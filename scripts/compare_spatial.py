"""Compare local history with one-ring context using ridge and logistic models.

Run only when compute is available: uv run python scripts/compare_spatial.py
Completed fits and exports are reused. No neural training or test evaluation.
"""

import argparse
import json
import logging
import resource
import time
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from compare_occurrence import (
    evaluate,
    read_probabilities,
    save_probabilities,
)

from atlas.baselines import write_json
from atlas.diagnosis import (
    read_predictions,
    save_predictions,
    save_tensors,
    scalar_score,
    score_breakdown,
)
from atlas.linear import (
    CONTROL_COLUMNS,
    Moments,
    control_inputs,
    historical_split,
    ridge_solutions,
)
from atlas.model import project_counts
from atlas.occurrence import logistic_fit, probability_scores
from atlas.samples import TARGET_NAMES, CellMonths, Preprocessing, WindowDataset
from atlas.spatial import NEIGHBOUR_NAMES, Neighbourhood
from atlas.splits import read_split
from atlas.training import CachedWindows, gpu_device, score_predictions

LOCAL_SIZE = len(CONTROL_COLUMNS["summary"])
VARIANTS = ("local", "neighbours")
FAMILIES = ("ridge", "logistic")


@torch.no_grad()
def cache_inputs(
    dataset: WindowDataset,
    neighbours: Neighbourhood,
    batch_size: int,
    device: torch.device,
) -> CachedWindows:
    inputs = torch.empty(
        (len(dataset), LOCAL_SIZE + len(NEIGHBOUR_NAMES)),
        device=device,
        dtype=torch.float64,
    )
    targets = torch.empty((len(dataset), 6, len(TARGET_NAMES)), device=device)
    mask = torch.empty_like(targets, dtype=torch.bool)
    for offset in range(0, len(dataset), batch_size):
        rows = np.arange(offset, min(offset + batch_size, len(dataset)))
        inputs[offset : offset + len(rows), :LOCAL_SIZE] = control_inputs(
            dataset, rows, device
        )[:, CONTROL_COLUMNS["summary"]]
        inputs[offset : offset + len(rows), LOCAL_SIZE:] = neighbours.summaries(
            dataset, rows, device
        )
        raw, available = dataset.target_batch(rows)
        targets[offset : offset + len(rows)] = torch.as_tensor(raw, device=device)
        mask[offset : offset + len(rows)] = torch.as_tensor(available, device=device)
    if not torch.isfinite(inputs).all():
        raise ValueError("Non-finite local or neighbouring inputs.")
    return CachedWindows(inputs, targets, mask)


def fit_models(
    cached: CachedWindows,
    folder: Path,
    strengths: dict[str, list[float]],
    batch_size: int,
) -> None:
    """Same observations and solvers on both sides of the information comparison."""
    if all(
        (folder / f"{family}-{s:g}.pt").exists()
        for family in FAMILIES
        for s in strengths[family]
    ):
        return
    if not cached.mask.all():
        raise ValueError(
            "This bounded ridge comparison requires complete training targets."
        )
    x = cached.inputs[:, :LOCAL_SIZE] if folder.name == "local" else cached.inputs
    moments = Moments.empty(x.shape[1], 6 * len(TARGET_NAMES), x.device)
    for offset in range(0, len(x), batch_size):
        target = cached.targets[offset : offset + batch_size].double()
        target = target.sign() * target.abs().log1p()
        moments.add(x[offset : offset + batch_size], target.flatten(1))
    mean = moments.total / moments.count
    scale = moments.covariance().diag().clamp_min(0).sqrt()
    scale = torch.where(scale < 1e-6, 1, scale)
    missing = [
        s for s in strengths["ridge"] if not (folder / f"ridge-{s:g}.pt").exists()
    ]
    started = time.monotonic()
    for strength, (weights, intercept) in zip(
        missing, ridge_solutions(moments, torch.diag(1 / scale), missing), strict=True
    ):
        save_tensors(
            folder / f"ridge-{strength:g}.pt",
            {
                "weights": weights,
                "intercept": intercept,
                "strength": strength,
                "training_samples": len(x),
                "input_size": x.shape[1],
                "parameters": weights.numel() + intercept.numel(),
                "seconds_shared_ridge_solve": time.monotonic() - started,
            },
        )
    del moments
    missing = [
        s for s in strengths["logistic"] if not (folder / f"logistic-{s:g}.pt").exists()
    ]
    if missing:
        standardized = (x - mean) / scale
        labels = (cached.targets.abs() >= 1).flatten(1)
        for strength in missing:
            print(
                f"Fitting {folder.parent.name}/{folder.name}/logistic-{strength:g}: "
                f"{len(x):,} windows.",
                flush=True,
            )
            started = time.monotonic()
            fitted = logistic_fit(
                standardized, labels, cached.mask.flatten(1), strength
            )
            save_tensors(
                folder / f"logistic-{strength:g}.pt",
                {
                    **fitted,
                    "mean": mean,
                    "scale": scale,
                    "training_samples": len(x),
                    "input_size": x.shape[1],
                    "parameters": (x.shape[1] + 1) * labels.shape[1],
                    "seconds": time.monotonic() - started,
                },
            )


@torch.no_grad()
def predict(path: Path, inputs: torch.Tensor, batch_size: int) -> torch.Tensor:
    fitted = torch.load(path, map_location=inputs.device, weights_only=True)
    probability = path.name.startswith("logistic-")
    result = torch.empty(
        (len(inputs), 6, len(TARGET_NAMES)), device=inputs.device, dtype=torch.float64
    )
    for offset in range(0, len(inputs), batch_size):
        x = inputs[offset : offset + batch_size, : fitted["input_size"]]
        if probability:
            x = (x - fitted["mean"]) / fitted["scale"]
        output = (x @ fitted["weights"] + fitted["intercept"]).reshape(
            -1, 6, len(TARGET_NAMES)
        )
        result[offset : offset + len(x)] = (
            output.sigmoid() if probability else project_counts(output)
        )
    return result


@torch.no_grad()
def evaluate_original(
    dataset: WindowDataset,
    cached: CachedWindows,
    folder: Path,
    group: str,
    selected: dict[str, dict[str, float]],
    raw: dict,
    paths: dict[str, Path],
    geo: dict,
) -> dict:
    """Reuse the existing magnitude, probability and fixed-inspection scoring."""
    parent = geo["parent_resolution"] if group == "geographic" else None
    parents = geo["validation_parents"] if group == "geographic" else None
    comparison = {}
    for variant in VARIANTS:
        for family in FAMILIES:
            label = f"{variant}-{family}"
            path = folder / f"{group}-{label}.parquet"
            metrics_path = path.with_suffix(".json")
            probability = family == "logistic"
            if not (path.exists() and metrics_path.exists()):
                if path.exists():
                    prediction = (
                        read_probabilities if probability else read_predictions
                    )(path, dataset, cached.inputs.device)
                else:
                    model = (
                        folder / variant / f"{family}-{selected[variant][family]:g}.pt"
                    )
                    prediction = predict(model, cached.inputs, raw["batch_size"])
                    if probability:
                        save_probabilities(path, dataset, prediction)
                    else:
                        save_predictions(path, dataset, prediction, raw["batch_size"])
                scores = evaluate(
                    dataset,
                    cached,
                    prediction if probability else prediction.abs(),
                    raw["inspection_budget"],
                    ranking_only=not probability,
                    parents=parents,
                    parent_resolution=parent,
                )
                if not probability:
                    scores["magnitude"] = score_breakdown(
                        dataset, prediction, parent, tuple(parents or ())
                    )
                write_json(metrics_path, scores)
                del prediction
            comparison[label] = json.loads(metrics_path.read_text())
    # Fixed, already measured references; these paths never train another model.
    for label in ("frequency", "recent", "trees", "logistic", "magnitude-ridge"):
        comparison[f"reference-{label}"] = json.loads(
            (paths["occurrence_run"] / f"{group}-{label}.json").read_text()
        )
    reference_ridge = json.loads(
        (paths["diagnosis_run"] / "original" / f"{group}-summary-1.json").read_text()
    )["scores"]
    reference_trees = json.loads(
        (paths["diagnosis_run"] / "original" / f"{group}-trees.json").read_text()
    )["scores"]
    comparison["reference-magnitude-ridge"]["magnitude"] = reference_ridge
    comparison["reference-magnitude-trees"] = {"magnitude": reference_trees}
    # Same local controls should reproduce previous measurements. Small solver
    # and feature precision differences are permitted, not changed populations.
    for family, reference, metric in (
        ("ridge", reference_ridge["overall"], "signed_log_rmse"),
        ("logistic", comparison["reference-logistic"]["overall"], "log_loss"),
    ):
        current = comparison[f"local-{family}"]
        current = (
            current["magnitude"]["overall"] if family == "ridge" else current["overall"]
        )
        if not np.isclose(current[metric], reference[metric], rtol=1e-5, atol=1e-8):
            raise ValueError(
                f"Local {family} does not reproduce the existing {group} comparator."
            )
        neighbour = comparison[f"neighbours-{family}"]
        neighbour_score = (
            neighbour["magnitude"]["overall"]
            if family == "ridge"
            else neighbour["overall"]
        )
        reduction = 100 * (1 - neighbour_score[metric] / current[metric])
        local = comparison[f"local-{family}"]
        inspection = "semantic_add_building"
        gain = 100 * (
            neighbour["inspection"][inspection][0]["precision"]
            - local["inspection"][inspection][0]["precision"]
        )
        ap_change = neighbour_score["average_precision"] - current["average_precision"]
        print(
            f"{group}/{family}: {reduction:+.3f}% {metric} reduction; "
            f"AP change {ap_change:+.6f}; "
            f"{gain:+.2f} building hits per 100 inspections.",
            flush=True,
        )
    return comparison


def run(config_path: Path) -> None:
    raw = tomllib.loads(config_path.read_text())
    if (
        raw["validation_years"] != [2020, 2021, 2022]
        or raw["ridge_strengths"] != [0.001, 0.01, 0.1, 1.0, 10.0]
        or raw["logistic_strengths"] != [0.0001, 0.001, 0.01, 0.1, 1.0]
        or raw["inspection_budget"] != 100
        or type(raw["batch_size"]) is not int
        or raw["batch_size"] < 1
    ):
        raise ValueError(
            "Use the declared spatial folds, penalties and inspection budget."
        )
    paths = {
        key: (config_path.parent / raw[key]).resolve()
        for key in (
            "dataset",
            "split",
            "preprocessing",
            "output",
            "diagnosis_run",
            "occurrence_run",
        )
    }
    split = read_split(paths["split"])
    if (split.input_months, split.target_months) != (24, 6):
        raise ValueError("This comparison retains the approved 24/6-month windows.")
    split = replace(
        split,
        months=tuple(m for m in split.months if m < split.temporal["validation"][1]),
    )
    resolved = {
        **raw,
        **{key: str(path) for key, path in paths.items()},
        "split_configuration": yaml.safe_load(paths["split"].read_text()),
        "neighbour_features": list(NEIGHBOUR_NAMES),
        "neighbour_policy": (
            "One ring excluding self; fixed domain and same train/validation group "
            "only; no buffers or test cells; observed months through cutoff only."
        ),
        "aggregation": (
            "Mean signed-log values and occurrence fractions over observed "
            "neighbour-months; state at cutoff; observed fractions and "
            "eligible/ring-size fraction."
        ),
        "selection": (
            "Separate penalty per input variant and objective, by mean "
            "2020/2021/2022 validation RMSE or log loss; ties follow declared order."
        ),
    }
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    configuration = output / "config.json"
    if configuration.exists() and json.loads(configuration.read_text()) != resolved:
        raise ValueError("Spatial configuration changed; use a new run directory.")
    if not configuration.exists():
        write_json(configuration, resolved)
    device = gpu_device()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    corpus = CellMonths.read(paths["dataset"], split)
    neighbours = Neighbourhood.from_split(split)
    fold_scores = {variant: {family: [] for family in FAMILIES} for variant in VARIANTS}
    for year in raw["validation_years"]:
        folder = output / str(year)
        folder.mkdir(exist_ok=True)
        fold = CellMonths(
            historical_split(split, year), corpus.values, corpus.available
        )
        prep_path = folder / "preprocessing.npz"
        if not prep_path.exists():
            Preprocessing.fit(fold).save(prep_path)
        prep = Preprocessing.read(prep_path)
        training = WindowDataset(fold, prep, "train")
        validation = WindowDataset(fold, prep, "validation")
        strengths = {family: raw[f"{family}_strengths"] for family in FAMILIES}
        pending = any(
            not (folder / variant / f"{family}-{s:g}.pt").exists()
            for variant in VARIANTS
            for family in FAMILIES
            for s in strengths[family]
        )
        cached = (
            cache_inputs(training, neighbours, raw["batch_size"], device)
            if pending
            else None
        )
        for variant in VARIANTS:
            destination = folder / variant
            destination.mkdir(exist_ok=True)
            if cached is not None:
                fit_models(cached, destination, strengths, raw["batch_size"])
        del cached
        pending = any(
            not (folder / variant / f"{family}-{s:g}.json").exists()
            for variant in VARIANTS
            for family in FAMILIES
            for s in strengths[family]
        )
        valid = (
            cache_inputs(validation, neighbours, raw["batch_size"], device)
            if pending
            else None
        )
        for variant in VARIANTS:
            for family in FAMILIES:
                scores = []
                for strength in strengths[family]:
                    path = folder / variant / f"{family}-{strength:g}.pt"
                    metrics_path = path.with_suffix(".json")
                    if not metrics_path.exists():
                        assert valid is not None
                        prediction = predict(path, valid.inputs, raw["batch_size"])
                        with torch.no_grad():
                            metrics = (
                                probability_scores(
                                    valid.targets,
                                    valid.mask,
                                    prediction,
                                    calibration=False,
                                )
                                if family == "logistic"
                                else score_predictions(validation, prediction)
                            )
                        write_json(metrics_path, metrics)
                        del prediction
                    metric = "log_loss" if family == "logistic" else "signed_log_rmse"
                    scores.append(
                        scalar_score(json.loads(metrics_path.read_text()), metric)
                    )
                fold_scores[variant][family].append(scores)
        del valid
        print(f"Completed historical fold {year}.", flush=True)
    selected = {
        variant: {
            family: raw[f"{family}_strengths"][
                int(np.array(fold_scores[variant][family]).mean(0).argmin())
            ]
            for family in FAMILIES
        }
        for variant in VARIANTS
    }
    write_json(
        output / "selection.json", {"fold_scores": fold_scores, "selected": selected}
    )
    print(f"Frozen historical selection: {selected}", flush=True)
    folder = output / "original"
    folder.mkdir(exist_ok=True)
    prep_path = folder / "preprocessing.npz"
    if not prep_path.exists():
        Preprocessing.read(paths["preprocessing"]).save(prep_path)
    prep = Preprocessing.read(prep_path)
    training = WindowDataset(corpus, prep, "train")
    pending = any(
        not (folder / variant / f"{family}-{selected[variant][family]:g}.pt").exists()
        for variant in VARIANTS
        for family in FAMILIES
    )
    cached = (
        cache_inputs(training, neighbours, raw["batch_size"], device)
        if pending
        else None
    )
    for variant in VARIANTS:
        destination = folder / variant
        destination.mkdir(exist_ok=True)
        if cached is not None:
            fit_models(
                cached,
                destination,
                {family: [selected[variant][family]] for family in FAMILIES},
                raw["batch_size"],
            )
    del cached
    comparison = {}
    geo = resolved["split_configuration"]["geographic"]
    for group, geographic in (("temporal", False), ("geographic", True)):
        dataset = WindowDataset(corpus, prep, "validation", geographic=geographic)
        cached = cache_inputs(dataset, neighbours, raw["batch_size"], device)
        comparison[group] = evaluate_original(
            dataset, cached, folder, group, selected, raw, paths, geo
        )
        del cached
    write_json(output / "comparison.json", comparison)
    write_json(
        output / "runtime.json"
        if not (output / "runtime.json").exists()
        else output / "resume-runtime.json",
        {
            "seconds_this_invocation": time.monotonic() - started,
            "peak_process_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "input_sizes": {
                "local": LOCAL_SIZE,
                "neighbours": LOCAL_SIZE + len(NEIGHBOUR_NAMES),
            },
        },
    )
    print(f"Spatial comparison complete: {output}", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/spatial.toml"))
    run(parser.parse_args().config)
