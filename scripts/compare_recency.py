"""Select recency on historical folds, refit ridge, and prepare paired MLP configs.

Run through Compose: uv run python scripts/compare_recency.py
Then use atlas train on the two generated TOML files. Completed fits and
evaluations are reused; the reserved-test period is never loaded.
"""

import argparse
import json
import resource
import time
import tomllib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import yaml

from atlas.baselines import write_json
from atlas.diagnosis import evaluate_controls, prediction_path, save_tensors
from atlas.linear import (
    CONTROL_COLUMNS,
    JOINT_SIZE,
    Moments,
    control_inputs,
    historical_split,
    recency_weights,
    ridge_solutions,
)
from atlas.metrics import transform
from atlas.samples import CellMonths, Preprocessing, WindowDataset
from atlas.splits import read_split
from atlas.training import gpu_device


def fit_summary(
    training: WindowDataset,
    directory: Path,
    policies: dict[str, int | None],
    strengths: dict[str, list[float]],
    batch_size: int,
    device: torch.device,
) -> None:
    """Share input scaling across policies; reweight only regression moments."""
    if all((directory / label / "summary.pt").exists() for label in policies):
        return
    columns = CONTROL_COLUMNS["summary"]
    path = directory / "moments.pt"
    if path.exists():
        moments = {
            name: Moments(**values)
            for name, values in torch.load(
                path, map_location=device, weights_only=True
            ).items()
        }
    else:
        moments = {name: Moments.empty(len(columns), 222, device) for name in policies}
        weights = {
            name: torch.as_tensor(recency_weights(training, half_life), device=device)
            if half_life is not None
            else None
            for name, half_life in policies.items()
        }
        for offset in range(0, len(training), batch_size):
            rows = np.arange(offset, min(offset + batch_size, len(training)))
            raw, mask = training.target_batch(rows)
            if not mask.all():
                raise ValueError(
                    "The bounded recency comparison requires complete training targets."
                )
            x = control_inputs(training, rows, device)[:, columns]
            y = transform(torch.as_tensor(raw, device=device).double()).flatten(1)
            for name, values in moments.items():
                weight = weights[name]
                values.add(
                    x,
                    y,
                    weight[offset : offset + len(rows)] if weight is not None else None,
                )
        save_tensors(path, {name: asdict(value) for name, value in moments.items()})
    # No weighted normalization or PCA: every policy uses this uniform covariance.
    scale = moments["uniform"].covariance().diag().clamp_min(0).sqrt()
    projection = torch.diag(1 / torch.where(scale < 1e-6, 1, scale))
    for name, observed in moments.items():
        destination = directory / name
        destination.mkdir(exist_ok=True)
        if (destination / "summary.pt").exists():
            continue
        weights = torch.zeros(
            (len(strengths[name]), JOINT_SIZE, 222), device=device, dtype=torch.float64
        )
        intercepts = weights.new_zeros((len(strengths[name]), 222))
        for index, (w, b) in enumerate(
            ridge_solutions(observed, projection, strengths[name])
        ):
            weights[index, columns] = w
            intercepts[index] = b
        save_tensors(
            destination / "summary.pt",
            {
                "weights": weights,
                "intercepts": intercepts,
                "strengths": strengths[name],
                "projection": projection,
                "training_samples": len(training),
                "half_life_months": policies[name],
            },
        )
    print(
        f"Fitted {directory.name}: {len(training):,} windows, {list(policies)}.",
        flush=True,
    )


def fold_score(directory: Path, strength: float) -> float:
    path = prediction_path(directory, "temporal", f"summary-{strength:g}")
    return float(
        json.loads(path.with_suffix(".json").read_text())["scores"]["overall"][
            "signed_log_rmse"
        ]
    )


@torch.inference_mode()
def compare(config_path: Path) -> None:
    started = time.monotonic()
    device = gpu_device()
    raw = tomllib.loads(config_path.read_text())
    if (
        raw["half_lives_months"] != [12, 24]
        or raw["ridge_strengths"] != [0.001, 0.01, 0.1, 1.0, 10.0]
        or raw["validation_years"] != [2020, 2021, 2022]
        or type(raw["batch_size"]) is not int
        or raw["batch_size"] < 1
    ):
        raise ValueError(
            "Use the approved recency policies, penalties, and historical folds."
        )
    paths = {
        key: (config_path.parent / raw[key]).resolve()
        for key in ("dataset", "split", "preprocessing", "output", "mlp_template")
    }
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    resolved = {
        **raw,
        **{k: str(v) for k, v in paths.items()},
        "split_configuration": yaml.safe_load(paths["split"].read_text()),
        "mlp_configuration": tomllib.loads(paths["mlp_template"].read_text()),
        "selection": (
            "Minimum mean historical-fold RMSE; ties follow half-life then penalty "
            "order. Select the best nonuniform policy for the neural comparison "
            "even if uniform wins."
        ),
        "weights": (
            "2**(-age_in_months/half_life), age from latest eligible training "
            "window's final target month; global training mean one; "
            "unweighted preprocessing and validation."
        ),
    }
    config_file = output / "config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != resolved:
        raise ValueError("Recency configuration changed; use a new run directory.")
    if not config_file.exists():
        write_json(config_file, resolved)
    split = read_split(paths["split"])
    split = replace(
        split,
        months=tuple(m for m in split.months if m < split.temporal["validation"][1]),
    )
    corpus = CellMonths.read(paths["dataset"], split)
    policies: dict[str, int | None] = {"uniform": None, "half-12": 12, "half-24": 24}
    strengths = raw["ridge_strengths"]
    scores: dict[str, list[list[float]]] = {name: [] for name in policies}
    for year in raw["validation_years"]:
        directory = output / str(year)
        directory.mkdir(exist_ok=True)
        fold = CellMonths(
            historical_split(split, year), corpus.values, corpus.available
        )
        prep_path = directory / "preprocessing.npz"
        if prep_path.exists():
            preprocessing = Preprocessing.read(prep_path)
        else:
            preprocessing = Preprocessing.fit(fold)
            preprocessing.save(prep_path)
        training = WindowDataset(fold, preprocessing, "train")
        validation = WindowDataset(fold, preprocessing, "validation")
        write_json(
            directory / "config.json",
            {
                "training_targets": [
                    d.isoformat() for d in fold.split.temporal["train"]
                ],
                "validation_targets": [
                    d.isoformat() for d in fold.split.temporal["validation"]
                ],
                "training_samples": len(training),
                "validation_samples": len(validation),
                "last_training_input": fold.split.months[
                    training.starts[-1] - 1
                ].isoformat(),
            },
        )
        fit_summary(
            training,
            directory,
            policies,
            {name: strengths for name in policies},
            raw["batch_size"],
            device,
        )
        for name in policies:
            evaluate_controls(
                validation,
                directory / name,
                "temporal",
                {"summary": strengths},
                raw["batch_size"],
                device,
            )
            scores[name].append(
                [fold_score(directory / name, strength) for strength in strengths]
            )
    chosen = {
        name: strengths[int(np.asarray(values).mean(0).argmin())]
        for name, values in scores.items()
    }
    means = {
        name: float(np.asarray(values).mean(0).min()) for name, values in scores.items()
    }
    nonuniform = min(["half-12", "half-24"], key=means.__getitem__)
    selection = {
        "fold_rmse": scores,
        "strengths": strengths,
        "selected_strengths": chosen,
        "mean_rmse": means,
        "best_policy": min(means, key=means.__getitem__),
        "selected_nonuniform": nonuniform,
        "half_life_months": policies[nonuniform],
    }
    write_json(output / "selection.json", selection)
    print(f"Frozen historical selection: {selection}", flush=True)
    directory = output / "original"
    directory.mkdir(exist_ok=True)
    preprocessing = Preprocessing.read(paths["preprocessing"])
    if not (directory / "preprocessing.npz").exists():
        preprocessing.save(directory / "preprocessing.npz")
    training = WindowDataset(corpus, preprocessing, "train")
    selected = {name: policies[name] for name in ("uniform", nonuniform)}
    fit_summary(
        training,
        directory,
        selected,
        {name: [chosen[name]] for name in selected},
        raw["batch_size"],
        device,
    )
    geo = resolved["split_configuration"]["geographic"]
    for group, geographic in (("temporal", False), ("geographic", True)):
        validation = WindowDataset(
            corpus, preprocessing, "validation", geographic=geographic
        )
        for name in selected:
            evaluate_controls(
                validation,
                directory / name,
                group,
                {"summary": [chosen[name]]},
                raw["batch_size"],
                device,
                geo["parent_resolution"] if geographic else None,
                tuple(geo["validation_parents"]) if geographic else (),
            )
    body = paths["mlp_template"].read_text().split("[model]", 1)[1]
    for name, half_life in selected.items():
        lines = [
            f"{key} = {json.dumps(str(paths[key]))}"
            for key in ("dataset", "split", "preprocessing")
        ]
        lines.append(f"output = {json.dumps(str(output / ('mlp-' + name)))}")
        if half_life is not None:
            lines.append(f"recency_half_life_months = {half_life}")
        contents = "\n".join(lines) + "\n\n[model]" + body
        path = output / f"mlp-{name}.toml"
        if path.exists() and path.read_text() != contents:
            raise ValueError(
                "Generated neural configuration differs from its saved version."
            )
        if not path.exists():
            path.write_text(contents)
    if not (output / "runtime.json").exists():
        write_json(
            output / "runtime.json",
            {
                "linear_seconds": time.monotonic() - started,
                "peak_process_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024,
                "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device)
                / 2**20,
            },
        )
    print(
        "Linear comparison complete. Train the two generated MLP configurations.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/recency.toml"))
    compare(parser.parse_args().config)
