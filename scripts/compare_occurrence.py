"""Bounded occurrence comparison. Run through Compose with uv run python.

Completed logistic fits, tree models, and neural epochs survive interruption.
All stages load development months only; no reserved-test evaluation is exposed.
"""

import argparse
import ctypes
import json
import resource
import time
import tomllib
from dataclasses import replace
from pathlib import Path

import h3
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from atlas.baselines import write_json
from atlas.diagnosis import read_predictions, save_tensors
from atlas.linear import historical_split
from atlas.model import ResidualMLP
from atlas.occurrence import logistic_fit, probability_scores, top_budget
from atlas.samples import TARGET_NAMES, CellMonths, Preprocessing, WindowDataset
from atlas.splits import read_split
from atlas.training import CachedWindows, cache_windows, gpu_device, sample_keys, train
from atlas.trees import RegressionTrees


def save_probabilities(
    path: Path, dataset: WindowDataset, values: torch.Tensor
) -> None:
    schema = pa.schema(
        [
            ("cell", pa.string()),
            ("cutoff", pa.string()),
            ("probability", pa.list_(pa.float64(), 222)),
        ],
        metadata={
            "target_names": json.dumps(TARGET_NAMES),
            "forecast_order": "horizon-major; months 1 through 6",
            "prediction_units": "P(abs(raw target) >= 1)",
        },
    )
    temporary = path.with_suffix(".partial")
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for offset in range(0, len(dataset), 4096):
            rows = np.arange(offset, min(offset + 4096, len(dataset)))
            writer.write_table(
                pa.Table.from_pydict(
                    {
                        **sample_keys(dataset, rows),
                        "probability": pa.FixedSizeListArray.from_arrays(
                            pa.array(
                                values[offset : offset + len(rows)]
                                .double()
                                .cpu()
                                .numpy()
                                .ravel()
                            ),
                            222,
                        ),
                    },
                    schema=schema,
                )
            )
    temporary.replace(path)


def read_probabilities(
    path: Path, dataset: WindowDataset, device: torch.device
) -> torch.Tensor:
    table = pq.read_table(path, columns=["cell", "cutoff", "probability"])
    keys = sample_keys(dataset, np.arange(len(dataset)))
    if any(table[name].to_pylist() != values for name, values in keys.items()):
        raise ValueError("Probability keys differ from evaluation rows.")
    if json.loads((table.schema.metadata or {})[b"target_names"]) != list(TARGET_NAMES):
        raise ValueError("Probability target names differ from the experiment.")
    return (
        torch.as_tensor(
            table["probability"].combine_chunks().values.to_numpy().copy(),
            device=device,
        )
        .double()
        .reshape(-1, 6, 37)
    )


def prepare(
    corpus: CellMonths, preprocessing: Preprocessing, device: torch.device
) -> tuple[WindowDataset, CachedWindows]:
    dataset = WindowDataset(corpus, preprocessing, "train")
    return dataset, cache_windows(
        ResidualMLP("summary", 205).to(device), dataset, device
    )


@torch.no_grad()
def logistic_predictions(fit: dict, inputs: torch.Tensor) -> torch.Tensor:
    result = torch.empty((len(inputs), 222), dtype=torch.float64, device=inputs.device)
    for offset in range(0, len(inputs), 32768):
        x = (inputs[offset : offset + 32768].double() - fit["mean"]) / fit["scale"]
        result[offset : offset + len(x)] = (
            x @ fit["weights"] + fit["intercept"]
        ).sigmoid()
    return result.reshape(-1, 6, 37)


def fit_linear(
    corpus: CellMonths, config: dict, paths: dict[str, Path], device: torch.device
) -> None:
    scores = []
    for year in [*config["validation_years"], None]:
        folder = paths["output"] / (str(year) if year else "original")
        folder.mkdir(exist_ok=True)
        fold = (
            CellMonths(
                historical_split(corpus.split, year), corpus.values, corpus.available
            )
            if year
            else corpus
        )
        prep_path = folder / "preprocessing.npz"
        if not prep_path.exists():
            prep = (
                Preprocessing.fit(fold)
                if year
                else Preprocessing.read(paths["preprocessing"])
            )
            prep.save(prep_path)
        prep = Preprocessing.read(prep_path)
        if year:
            strengths = config["logistic_strengths"]
        else:
            means = np.array(scores).mean(0)
            selected = config["logistic_strengths"][int(means.argmin())]
            write_json(
                paths["output"] / "selection.json",
                {
                    "fold_log_loss": scores,
                    "mean_log_loss": means.tolist(),
                    "strengths": config["logistic_strengths"],
                    "strength": selected,
                },
            )
            print(
                f"Selected logistic strength {selected:g} from historical folds.",
                flush=True,
            )
            strengths = [selected]
        if all(
            (folder / f"logistic-{s:g}.pt").exists()
            and (not year or (folder / f"logistic-{s:g}.json").exists())
            for s in strengths
        ):
            if year:
                scores.append(
                    [
                        json.loads((folder / f"logistic-{s:g}.json").read_text())[
                            "log_loss"
                        ]
                        for s in strengths
                    ]
                )
            continue
        training, cached = prepare(fold, prep, device)
        validation = WindowDataset(fold, prep, "validation")
        valid = (
            cache_windows(ResidualMLP("summary", 205).to(device), validation, device)
            if year
            else None
        )
        mean = cached.inputs.double().mean(0)
        scale = cached.inputs.double().std(0, correction=0)
        scale = torch.where(scale < 1e-6, 1, scale)
        x = (cached.inputs.double() - mean) / scale
        labels = (cached.targets.abs() >= 1).flatten(1)
        mask = cached.mask.flatten(1)
        if not (folder / "frequency.pt").exists():
            save_tensors(
                folder / "frequency.pt",
                {
                    "probability": ((labels & mask).sum(0).double() + 0.5)
                    / (mask.sum(0) + 1)
                },
            )
        fold_scores = []
        for strength in strengths:
            path = folder / f"logistic-{strength:g}.pt"
            started = time.monotonic()
            if path.exists():
                fit = torch.load(path, map_location=device, weights_only=True)
            else:
                fit = {
                    **logistic_fit(x, labels, mask, strength),
                    "mean": mean,
                    "scale": scale,
                    "training_samples": len(training),
                    "seconds": time.monotonic() - started,
                }
                save_tensors(path, fit)
            if valid is not None:
                probabilities = logistic_predictions(fit, valid.inputs)
                metrics = probability_scores(
                    valid.targets, valid.mask, probabilities, calibration=False
                )
                write_json(path.with_suffix(".json"), metrics)
                fold_scores.append(metrics["log_loss"])
            print(
                f"Logistic {folder.name}, lambda={strength:g}: "
                f"{fit['iterations']} iterations; gradient "
                f"{fit['max_gradient']:.3g}; {fit['seconds']:.1f} s",
                flush=True,
            )
        if year:
            scores.append(fold_scores)
        del cached, valid, x, labels, mask


def fit_trees(
    training: CachedWindows,
    validation: CachedWindows,
    config: dict,
    folder: Path,
    device: torch.device,
) -> None:
    """Binary OpenCL training; raw numeric tree sums are logits on GPU."""
    ctypes.CDLL(
        str(Path(torch.__file__).parent / "lib/libhsa-runtime64.so"),
        mode=ctypes.RTLD_GLOBAL,
    )
    params = {
        **config["trees"],
        "objective": "binary",
        "metric": "binary_logloss",
        "device_type": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "gpu_use_dp": True,
        "num_threads": 6,
        "max_bin": 63,
        "bin_construct_sample_cnt": 20000,
        "feature_pre_filter": False,
        "zero_as_missing": False,
        "seed": config["seed"],
        "data_random_seed": config["seed"],
        "verbosity": -1,
    }
    folder.mkdir(exist_ok=True)
    data = lgb.Dataset(training.inputs.cpu().numpy(), params=params).construct()
    valid = lgb.Dataset(
        validation.inputs.cpu().numpy(), params=params, reference=data
    ).construct()
    for h in range(6):
        for c, name in enumerate(TARGET_NAMES):
            path = folder / f"h{h + 1}-{name}.txt"
            constant = path.with_suffix(".json")
            if path.exists() or constant.exists():
                continue
            started = time.monotonic()
            observed = training.mask[:, h, c]
            hits = (training.targets[:, h, c].abs() >= 1) & observed
            count = int(observed.sum())
            if not count:
                raise ValueError("No observed tree training labels.")
            positive = int(hits.sum())
            if positive in (0, count):
                write_json(constant, {"probability": (positive + 0.5) / (count + 1)})
                continue
            for binned, cached in ((data, training), (valid, validation)):
                binned.set_label(
                    (cached.targets[:, h, c].abs() >= 1).float().cpu().numpy()
                )
                binned.set_weight(cached.mask[:, h, c].float().cpu().numpy())
            model = lgb.train(
                params,
                data,
                num_boost_round=config["rounds"],
                valid_sets=[valid],
                callbacks=[
                    lgb.early_stopping(config["early_stopping_rounds"], verbose=False)
                ],
            )
            text = model.model_to_string()
            # Manual acceptance of the GPU evaluator against LightGBM's raw
            # predictions. Host prediction here is a bounded correctness check.
            rows = np.linspace(0, len(validation.inputs) - 1, 64, dtype=int)
            probe = validation.inputs[torch.as_tensor(rows, device=device)]
            with torch.inference_mode():
                actual = RegressionTrees(text).to(device)(probe)
            expected = torch.as_tensor(
                model.predict(probe.cpu().numpy(), raw_score=True), device=device
            )
            torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-8)
            temporary = path.with_suffix(".partial")
            temporary.write_text(text)
            temporary.replace(path)
            print(
                f"Probability tree {h * 37 + c + 1}/222: "
                f"{model.best_iteration} rounds; {time.monotonic() - started:.1f} s",
                flush=True,
            )


@torch.inference_mode()
def tree_predictions(folder: Path, inputs: torch.Tensor) -> torch.Tensor:
    predictions = torch.empty(
        (len(inputs), 6, 37), device=inputs.device, dtype=torch.float64
    )
    for h in range(6):
        for c, name in enumerate(TARGET_NAMES):
            path = folder / f"h{h + 1}-{name}.txt"
            if path.with_suffix(".json").exists():
                predictions[:, h, c] = json.loads(
                    path.with_suffix(".json").read_text()
                )["probability"]
                continue
            forest = RegressionTrees(path.read_text()).to(inputs.device)
            for offset in range(0, len(inputs), 4096):
                predictions[offset : offset + 4096, h, c] = forest(
                    inputs[offset : offset + 4096]
                ).sigmoid()
        print(f"Predicted probability trees: horizon {h + 1}/6.", flush=True)
    return predictions


@torch.inference_mode()
def evaluate(
    dataset: WindowDataset,
    cached: CachedWindows,
    predictions: torch.Tensor,
    budget: int,
    *,
    ranking_only: bool = False,
    parents: list[str] | None = None,
    parent_resolution: int | None = None,
) -> dict[str, object]:
    """Natural distributions, actual target-year masks, and fixed-budget rankings."""
    device = predictions.device
    result: dict[str, object] = {
        "overall": probability_scores(
            cached.targets, cached.mask, predictions, ranking_only=ranking_only
        )
    }
    years = np.array([m.year for m in dataset.corpus.split.months])
    target_year = years[
        np.tile(dataset.starts, len(dataset.positions))[:, None] + np.arange(6)
    ]
    for year in np.unique(target_year):
        mask = (
            cached.mask
            & torch.as_tensor(target_year == year, device=device)[:, :, None]
        )
        result[f"year/{year}"] = probability_scores(
            cached.targets,
            mask,
            predictions,
            ranking_only=ranking_only,
            calibration=False,
        )
    if parents is not None:
        if parent_resolution is None:
            raise ValueError(
                "Geographic probability scores need the parent resolution."
            )
        ids = np.repeat(
            [
                h3.cell_to_parent(dataset.corpus.split.cells[p], parent_resolution)
                for p in dataset.positions
            ],
            len(dataset.starts),
        )
        for parent in parents:
            selected = torch.as_tensor(ids == parent, device=device)
            result[f"parent/{parent}"] = probability_scores(
                cached.targets[selected],
                cached.mask[selected],
                predictions[selected],
                ranking_only=ranking_only,
                calibration=False,
            )
    inspection = {}
    for c, name in enumerate(TARGET_NAMES):
        horizons = []
        for h in range(6):
            inspected, hits, support, random_hits = 0, 0.0, 0, 0.0
            for cutoff in range(len(dataset.starts)):
                rows = torch.arange(
                    cutoff, len(dataset), len(dataset.starts), device=device
                )
                rows = rows[cached.mask[rows, h, c]]
                y = cached.targets[rows, h, c].abs() >= 1
                scores = predictions[rows, h, c]
                top = top_budget(y, scores, budget)
                inspected += int(top["inspected"])
                hits += float(top["expected_hits"])
                positives = int(y.sum())
                support += positives
                random_hits += min(budget, len(rows)) * positives / max(1, len(rows))
            horizons.append(
                {
                    "inspected": inspected,
                    "expected_hits": hits,
                    "positive": support,
                    "precision": hits / inspected if inspected else None,
                    "recall": hits / support if support else None,
                    "random_expected_hits": random_hits,
                    "lift_over_random": hits / random_hits if random_hits else None,
                }
            )
        inspection[name] = horizons
    result["inspection"] = inspection
    return result


def run(config_path: Path, stage: str) -> None:
    started = time.monotonic()
    device = gpu_device()
    raw = tomllib.loads(config_path.read_text())
    if (
        raw["validation_years"] != [2020, 2021, 2022]
        or raw["logistic_strengths"] != [0.0001, 0.001, 0.01, 0.1, 1.0]
        or raw["inspection_budget"] != 100
    ):
        raise ValueError(
            "Use the declared occurrence folds, penalties, and inspection budget."
        )
    paths = {
        key: (config_path.parent / raw[key]).resolve()
        for key in (
            "dataset",
            "split",
            "preprocessing",
            "output",
            "mlp_template",
            "tree_template",
            "diagnosis_run",
            "magnitude_mlp",
        )
    }
    tree_config = tomllib.loads(paths["tree_template"].read_text())
    output = paths["output"]
    output.mkdir(exist_ok=True)
    resolved = {
        **raw,
        **{k: str(v) for k, v in paths.items()},
        "split_configuration": yaml.safe_load(paths["split"].read_text()),
        "mlp_template": tomllib.loads(paths["mlp_template"].read_text()),
        "tree_template": tree_config,
        "event": "abs(raw target) >= 1",
        "probability_clip": [1e-7, 1 - 1e-7],
        "logistic_solver": {
            "method": "full-data float64 L-BFGS with output curvature scaling",
            "max_iterations": 1000,
            "maximum_accepted_gradient": 5e-5,
        },
    }
    config_file = output / "config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != resolved:
        raise ValueError("Occurrence configuration changed; use a new run directory.")
    if not config_file.exists():
        write_json(config_file, resolved)
    split = read_split(paths["split"])
    split = replace(
        split,
        months=tuple(m for m in split.months if m < split.temporal["validation"][1]),
    )
    corpus = CellMonths.read(paths["dataset"], split)
    if stage in ("all", "linear"):
        fit_linear(corpus, raw, paths, device)
    # Reuse atlas train rather than add a second neural training loop.
    mlp_config = output / "mlp.toml"
    body = paths["mlp_template"].read_text().split("[model]", 1)[1]
    contents = "\n".join(
        f"{key} = {json.dumps(str(paths[key]))}"
        for key in ("dataset", "split", "preprocessing")
    )
    contents += (
        f"\noutput = {json.dumps(str(output / 'mlp'))}"
        '\nobjective = "occurrence"\n\n[model]' + body
    )
    if mlp_config.exists() and mlp_config.read_text() != contents:
        raise ValueError("Generated MLP configuration changed.")
    if not mlp_config.exists():
        mlp_config.write_text(contents)
    if stage == "linear":
        return
    if stage == "all" and not all(
        (output / "mlp" / name).exists()
        for name in ("metrics.json", "temporal.parquet", "geographic.parquet")
    ):
        checkpoint = output / "mlp/latest.pt"
        train(mlp_config, checkpoint if checkpoint.exists() else None)
    prep = Preprocessing.read(paths["preprocessing"])
    datasets = {
        name: WindowDataset(corpus, prep, "validation", geographic=geo)
        for name, geo in (("temporal", False), ("geographic", True))
    }
    cache_model = ResidualMLP("summary", 205).to(device)
    cached = {
        name: cache_windows(cache_model, dataset, device)
        for name, dataset in datasets.items()
    }
    if stage in ("all", "trees"):
        _, training = prepare(corpus, prep, device)
        fit_trees(training, cached["temporal"], tree_config, output / "trees", device)
        del training
    if stage == "trees":
        return
    selected = json.loads((output / "selection.json").read_text())["strength"]
    fit = torch.load(
        output / "original" / f"logistic-{selected:g}.pt",
        map_location=device,
        weights_only=True,
    )
    prior = torch.load(
        output / "original/frequency.pt", map_location=device, weights_only=True
    )["probability"].reshape(6, 37)
    summaries = {}
    for group, dataset in datasets.items():
        values = cached[group]
        models = [
            "frequency",
            "recent",
            "logistic",
            "trees",
            "mlp",
            "magnitude-trees",
            "magnitude-ridge",
            "magnitude-mlp",
        ]
        summaries[group] = {}
        for name in models:
            path = output / f"{group}-{name}.parquet"
            score_path = path.with_suffix(".json")
            if score_path.exists() and (path.exists() or name.startswith("magnitude-")):
                summaries[group][name] = json.loads(score_path.read_text())["overall"]
                continue
            ranking = name.startswith("magnitude-")
            if name == "frequency":
                probabilities = prior.expand(len(dataset), -1, -1)
            elif name == "recent":
                # Summary order: 30 final-state/calendar/mask inputs, then
                # 37 means, 37 active fractions, and 37 observed fractions.
                count = values.inputs[:, 104:141].double() * 6
                hits = values.inputs[:, 67:104].double() * count
                probabilities = (hits[:, None, :] + prior) / (count[:, None, :] + 1)
            elif name == "logistic":
                probabilities = logistic_predictions(fit, values.inputs)
            elif name == "trees":
                probabilities = tree_predictions(output / "trees", values.inputs)
            elif name == "mlp":
                probabilities = read_probabilities(
                    output / "mlp" / f"{group}.parquet", dataset, device
                )
            else:
                source = {
                    "magnitude-trees": paths["diagnosis_run"]
                    / "original"
                    / f"{group}-trees.parquet",
                    "magnitude-ridge": paths["diagnosis_run"]
                    / "original"
                    / f"{group}-summary-1.parquet",
                    "magnitude-mlp": paths["magnitude_mlp"] / f"{group}.parquet",
                }[name]
                probabilities = read_predictions(source, dataset, device).abs()
            metrics = evaluate(
                dataset,
                values,
                probabilities,
                raw["inspection_budget"],
                ranking_only=ranking,
                parents=resolved["split_configuration"]["geographic"][
                    "validation_parents"
                ]
                if group == "geographic"
                else None,
                parent_resolution=resolved["split_configuration"]["geographic"][
                    "parent_resolution"
                ]
                if group == "geographic"
                else None,
            )
            if not ranking:
                save_probabilities(path, dataset, probabilities)
            write_json(score_path, metrics)
            overall = metrics["overall"]
            assert isinstance(overall, dict)
            summaries[group][name] = overall
            print(
                f"{group} {name}: AP={overall['average_precision']:.6f}; "
                f"log loss={overall.get('log_loss')}",
                flush=True,
            )
        del probabilities
    write_json(output / "comparison.json", summaries)
    if not (output / f"runtime-{stage}.json").exists():
        write_json(
            output / f"runtime-{stage}.json",
            {
                "seconds": time.monotonic() - started,
                "peak_process_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024,
                "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device)
                / 2**20,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/occurrence.toml"))
    parser.add_argument(
        "--stage", choices=["all", "linear", "trees", "evaluate"], default="all"
    )
    args = parser.parse_args()
    run(args.config, args.stage)
