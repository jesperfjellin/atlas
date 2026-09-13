"""Run the approved, bounded development baseline-and-diagnosis comparison."""

import json
import resource
import time
import tomllib
from dataclasses import asdict, replace
from pathlib import Path

import h3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from atlas.baselines import gpu_inputs, recent_rate, target_column, write_json
from atlas.linear import (
    CONTROL_COLUMNS,
    JOINT_SIZE,
    Moments,
    accumulate,
    change_history,
    control_inputs,
    historical_split,
    history_summaries,
    representation_maps,
    ridge_solutions,
    target_groups,
)
from atlas.metrics import Scores, score_column
from atlas.model import project_counts
from atlas.samples import TARGET_NAMES, CellMonths, Preprocessing, WindowDataset
from atlas.splits import read_split
from atlas.training import gpu_device, sample_keys
from atlas.trees import RegressionTrees


def save_tensors(path: Path, value: object) -> None:
    temporary = path.with_suffix(".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def fit_controls(
    training: WindowDataset,
    directory: Path,
    strengths: dict[str, list[float]],
    dimensions: int,
    batch_size: int,
    device: torch.device,
) -> None:
    """Reuse completed fits; no optimizer or general scheduling machinery."""
    if all((directory / f"{name}.pt").exists() for name in CONTROL_COLUMNS):
        print(f"Reusing all fits in {directory.name}.", flush=True)
        return
    started = time.monotonic()
    path = directory / "moments.pt"
    if path.exists():
        moments = Moments(**torch.load(path, map_location=device, weights_only=True))
    else:
        moments = accumulate(training, device, batch_size)
        torch.cuda.synchronize()
        save_tensors(path, asdict(moments))
        write_json(
            directory / "accumulation.json",
            {
                "training_samples": moments.count,
                "seconds": time.monotonic() - started,
                "precision": "float64 sums and matrix products on AMD GPU",
            },
        )
    maps = representation_maps(moments, dimensions)
    groups = target_groups(training)
    missing = [
        name for name in CONTROL_COLUMNS if not (directory / f"{name}.pt").exists()
    ]
    weights = {
        name: torch.zeros(
            (len(strengths[name]), JOINT_SIZE, 222), device=device, dtype=torch.float64
        )
        for name in missing
    }
    intercepts = {
        name: value.new_zeros((len(strengths[name]), 222))
        for name, value in weights.items()
    }
    # Usually one complete mask. Different masks use separate, bounded passes;
    # never reuse a covariance from rows where an output was unavailable.
    for columns, rows in groups:
        if not len(rows):
            continue
        observed = (
            moments
            if len(rows) == len(training)
            else accumulate(training, device, batch_size, rows)
        )
        for name in missing:
            for index, (w, b) in enumerate(
                ridge_solutions(observed, maps[name], strengths[name])
            ):
                weights[name][index, :, columns] = w[:, columns]
                intercepts[name][index, columns] = b[columns]
    torch.cuda.synchronize()
    for name in missing:
        save_tensors(
            directory / f"{name}.pt",
            {
                "strengths": strengths[name],
                "weights": weights[name],
                "intercepts": intercepts[name],
                "projection": maps[name],
                "input_mean": moments.total / moments.count,
                "dimension": maps[name].shape[1],
                "training_samples": len(training),
            },
        )
    write_json(
        directory / "fitting.json",
        {
            "seconds_this_invocation": time.monotonic() - started,
            "target_mask_groups": len(groups),
            "dimensions": {name: value.shape[1] for name, value in maps.items()},
            "pca": (
                "Exact eigenvectors of full-training centered input covariance; "
                "64 components, whitened for ridge"
            ),
        },
    )
    print(
        f"Completed {directory.name} full-data fits: {len(training):,} rows.",
        flush=True,
    )


def prediction_path(directory: Path, group: str, name: str) -> Path:
    return directory / f"{group}-{name}.parquet"


def evaluation_complete(path: Path, parent_ids: tuple[str, ...]) -> bool:
    if not path.exists() or not path.with_suffix(".json").exists():
        return False
    scores = json.loads(path.with_suffix(".json").read_text())["scores"]
    return all(f"parent/{parent}" in scores for parent in parent_ids)


def save_predictions(
    path: Path, dataset: WindowDataset, predictions: torch.Tensor, batch_size: int
) -> None:
    schema = pa.schema(
        [
            ("cell", pa.string()),
            ("cutoff", pa.string()),
            ("prediction_log", pa.list_(pa.float64(), 222)),
        ],
        metadata={
            "target_names": json.dumps(TARGET_NAMES),
            "forecast_order": "horizon-major; months 1 through 6",
            "prediction_units": "signed_log1p; count forecasts projected to >= 0",
        },
    )
    temporary = path.with_suffix(".partial")
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for offset in range(0, len(dataset), batch_size):
            rows = np.arange(offset, min(offset + batch_size, len(dataset)))
            values = predictions[offset : offset + len(rows)].cpu().numpy().ravel()
            writer.write_table(
                pa.Table.from_pydict(
                    {
                        **sample_keys(dataset, rows),
                        "prediction_log": pa.FixedSizeListArray.from_arrays(
                            pa.array(values), 222
                        ),
                    },
                    schema=schema,
                )
            )
    temporary.replace(path)


def read_predictions(
    path: Path, dataset: WindowDataset, device: torch.device
) -> torch.Tensor:
    table = pq.read_table(path, columns=["cell", "cutoff", "prediction_log"])
    keys = sample_keys(dataset, np.arange(len(dataset)))
    if any(table[name].to_pylist() != values for name, values in keys.items()):
        raise ValueError(f"Forecast keys do not match validation rows: {path}")
    if json.loads((table.schema.metadata or {})[b"target_names"]) != list(TARGET_NAMES):
        raise ValueError("Forecast target columns differ from the experiment.")
    values = table["prediction_log"].combine_chunks().values.to_numpy().copy()
    return torch.as_tensor(values.reshape(-1, 6, 37), device=device).double()


def score_breakdown(
    dataset: WindowDataset,
    predictions: torch.Tensor,
    parent_resolution: int | None,
    parent_ids: tuple[str, ...] = (),
) -> dict[str, dict[str, object]]:
    """Use actual target year for each horizon, including cross-year windows."""
    device = predictions.device
    if predictions.shape != (len(dataset), 6, 37):
        raise ValueError("Forecast shape differs from validation population.")
    starts = np.tile(dataset.starts, len(dataset.positions))
    years = np.array([m.year for m in dataset.corpus.split.months])
    target_years = years[starts[:, None] + np.arange(6)]
    selectors = {"overall": np.ones((len(dataset), 6), dtype=bool)}
    unique_years = np.unique(target_years)
    if len(unique_years) > 1:
        selectors.update(
            {f"year/{year}": target_years == year for year in unique_years}
        )
    if parent_resolution is not None:
        parents = np.repeat(
            [
                h3.cell_to_parent(dataset.corpus.split.cells[p], parent_resolution)
                for p in dataset.positions
            ],
            len(dataset.starts),
        )
        selectors.update(
            {
                f"parent/{parent}": np.broadcast_to(
                    (parents == parent)[:, None], (len(dataset), 6)
                )
                for parent in sorted(set(parents) | set(parent_ids))
            }
        )
    tables = {name: Scores() for name in selectors}
    # Column-wise scoring retains the scorer's tied AP and missing-value rules.
    for h in range(6):
        for c, name in enumerate(TARGET_NAMES):
            raw, mask = (
                torch.as_tensor(v, device=device) for v in target_column(dataset, h, c)
            )
            for key, selected in selectors.items():
                rows = torch.as_tensor(selected[:, h].copy(), device=device)
                tables[key].record(
                    h,
                    c,
                    score_column(
                        raw[rows],
                        predictions[rows, h, c],
                        mask[rows],
                        count_target=not name.startswith("net_"),
                    ),
                )
    result = {
        key: {"samples": int(selectors[key].any(1).sum()), **table.summary()}
        for key, table in tables.items()
    }
    if len(unique_years) == 1:
        result[f"year/{unique_years[0]}"] = result["overall"]
    return result


def scalar_score(summary: dict[str, object], metric: str) -> float:
    value = summary[metric]
    if not isinstance(value, (float, int)) or not np.isfinite(value):
        raise ValueError(f"Unavailable aggregate {metric}.")
    return float(value)


def evaluate_controls(
    dataset: WindowDataset,
    directory: Path,
    group: str,
    strengths: dict[str, list[float]],
    batch_size: int,
    device: torch.device,
    parent_resolution: int | None = None,
    parent_ids: tuple[str, ...] = (),
) -> None:
    for name, candidates in strengths.items():
        fit = torch.load(
            directory / f"{name}.pt", map_location=device, weights_only=True
        )
        for strength in candidates:
            label = f"{name}-{strength:g}"
            path = prediction_path(directory, group, label)
            metrics_path = path.with_suffix(".json")
            if evaluation_complete(path, parent_ids):
                continue
            started = time.monotonic()
            if path.exists():
                predictions = read_predictions(path, dataset, device)
            else:
                index = fit["strengths"].index(strength)
                predictions = torch.empty(
                    (len(dataset), 6, 37), device=device, dtype=torch.float64
                )
                for offset in range(0, len(dataset), batch_size):
                    rows = np.arange(offset, min(offset + batch_size, len(dataset)))
                    x = control_inputs(dataset, rows, device)
                    predictions[offset : offset + len(rows)] = project_counts(
                        (x @ fit["weights"][index] + fit["intercepts"][index]).reshape(
                            -1, 6, 37
                        )
                    )
                save_predictions(path, dataset, predictions, batch_size)
            scores = score_breakdown(
                dataset, predictions, parent_resolution, parent_ids
            )
            write_json(
                metrics_path, {"scores": scores, "seconds": time.monotonic() - started}
            )
            print(
                f"{directory.name}/{group}/{label}: "
                f"{scalar_score(scores['overall'], 'signed_log_rmse'):.6f} RMSE, "
                f"{scalar_score(scores['overall'], 'average_precision'):.6f} AP",
                flush=True,
            )


def evaluate_simple(
    dataset: WindowDataset,
    directory: Path,
    group: str,
    batch_size: int,
    device: torch.device,
    parent_resolution: int | None = None,
    parent_ids: tuple[str, ...] = (),
) -> None:
    for name in ("zero", "recent_rate", "recent_log"):
        path = prediction_path(directory, group, name)
        if evaluation_complete(path, parent_ids):
            continue
        started = time.monotonic()
        if path.exists():
            predictions = read_predictions(path, dataset, device)
        else:
            if name == "zero":
                rate = torch.zeros(
                    (len(dataset), 37), device=device, dtype=torch.float64
                )
            elif name == "recent_rate":
                rate = recent_rate(dataset, 6, device).double()
            else:
                rate = torch.empty(
                    (len(dataset), 37), device=device, dtype=torch.float64
                )
                for offset in range(0, len(dataset), batch_size):
                    rows = np.arange(offset, min(offset + batch_size, len(dataset)))
                    values, mask = change_history(dataset, rows, device)
                    rate[offset : offset + len(rows)] = history_summaries(values, mask)[
                        :, :37
                    ]
            predictions = project_counts(rate[:, None].expand(-1, 6, -1))
            save_predictions(path, dataset, predictions, batch_size)
        scores = score_breakdown(dataset, predictions, parent_resolution, parent_ids)
        write_json(
            path.with_suffix(".json"),
            {"scores": scores, "seconds": time.monotonic() - started},
        )
        print(
            f"{directory.name}/{group}/{name}: "
            f"{scalar_score(scores['overall'], 'signed_log_rmse'):.6f} RMSE",
            flush=True,
        )


def evaluate_references(
    dataset: WindowDataset,
    directory: Path,
    group: str,
    tree_run: Path,
    gru_runs: list[Path],
    batch_size: int,
    device: torch.device,
    parent_resolution: int | None,
    parent_ids: tuple[str, ...] = (),
) -> None:
    """Score saved models only on original validation, never historical folds."""
    tree_path = prediction_path(directory, group, "trees")
    if not evaluation_complete(tree_path, parent_ids):
        started = time.monotonic()
        if tree_path.exists():
            predictions = read_predictions(tree_path, dataset, device)
        else:
            # Existing evaluator caches these ~2 GiB inputs; each GPU prediction
            # call remains bounded. No model is fitted or selected here.
            inputs = gpu_inputs(dataset, device)
            predictions = torch.empty(
                (len(dataset), 6, 37), device=device, dtype=torch.float64
            )
            for h in range(6):
                for c, name in enumerate(TARGET_NAMES):
                    forest = RegressionTrees(
                        (tree_run / "models" / f"h{h + 1}-{name}.txt").read_text()
                    ).to(device)
                    for offset in range(0, len(dataset), batch_size):
                        predictions[offset : offset + batch_size, h, c] = forest(
                            inputs[offset : offset + batch_size]
                        )
                print(f"Scored saved trees: {group}, horizon {h + 1}/6.", flush=True)
            del inputs
            predictions = project_counts(predictions)
            save_predictions(tree_path, dataset, predictions, batch_size)
        scores = score_breakdown(dataset, predictions, parent_resolution, parent_ids)
        original = json.loads((tree_run / "metrics.json").read_text())[group][
            "boosted_trees"
        ]
        for metric in ("signed_log_rmse", "average_precision"):
            if not np.isclose(
                scalar_score(scores["overall"], metric),
                original[metric],
                rtol=1e-7,
                atol=1e-9,
            ):
                raise ValueError(
                    "Saved tree predictions do not reproduce the reference scores."
                )
        write_json(
            tree_path.with_suffix(".json"),
            {"scores": scores, "seconds": time.monotonic() - started},
        )
    for index, run in enumerate(gru_runs):
        path = prediction_path(directory, group, f"gru-{index + 1}")
        if evaluation_complete(path, parent_ids):
            continue
        started = time.monotonic()
        predictions = read_predictions(run / f"{group}.parquet", dataset, device)
        scores = score_breakdown(dataset, predictions, parent_resolution, parent_ids)
        save_predictions(path, dataset, predictions, batch_size)
        write_json(
            path.with_suffix(".json"),
            {
                "source": str(run),
                "scores": scores,
                "seconds": time.monotonic() - started,
            },
        )


def building_diagnostics(corpus: CellMonths) -> dict[str, object]:
    """Unique development cell-months; no overlapping-window duplication."""
    from atlas.features import FEATURE_NAMES

    column = FEATURE_NAMES.index("semantic_add_building")
    result = {}
    for group, partition, geographic in (
        ("training", "train", False),
        ("temporal", "validation", False),
        ("geographic", "validation", True),
    ):
        eligible = set(corpus.split.eligible_cells(partition, geographic=geographic))
        cells = [i for i, cell in enumerate(corpus.split.cells) if cell in eligible]
        start, end = corpus.split.temporal[partition]
        for year in sorted({m.year for m in corpus.split.months if start <= m < end}):
            months = [
                i
                for i, m in enumerate(corpus.split.months)
                if start <= m < end and m.year == year
            ]
            values = corpus.values[np.ix_(cells, months, [column])].ravel()
            mask = corpus.available[np.ix_(cells, months, [column])].ravel()
            values = values[mask].astype(np.float64)
            positive = np.sort(values[values >= 1])
            total = float(values.sum())
            top_count = max(1, int(np.ceil(len(positive) * 0.01)))
            result[f"{group}/{year}"] = {
                "observed_cell_months": len(values),
                "positive_cell_months": len(positive),
                "positive_fraction": len(positive) / len(values)
                if len(values)
                else None,
                "positive_mean": float(positive.mean()) if len(positive) else None,
                "positive_median": float(np.median(positive))
                if len(positive)
                else None,
                "total_additions": total,
                "top_1_percent_positive_months_count": min(top_count, len(positive)),
                "top_1_percent_positive_months_share": float(
                    positive[-top_count:].sum() / total
                )
                if total
                else None,
                "largest_month_share": float(positive[-1] / total) if total else None,
            }
    return result


@torch.inference_mode()
def run_diagnosis(config_path: Path) -> None:
    device = gpu_device()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    raw = tomllib.loads(config_path.read_text())
    paths = {
        key: (config_path.parent / raw[key]).resolve()
        for key in ("dataset", "split", "preprocessing", "output", "tree_run")
    }
    gru_runs = [(config_path.parent / path).resolve() for path in raw["gru_runs"]]
    strengths = [float(v) for v in raw["ridge_strengths"]]
    if (
        not 1 <= len(strengths) <= 5
        or len(set(strengths)) != len(strengths)
        or any(not np.isfinite(v) or v <= 0 for v in strengths)
    ):
        raise ValueError(
            "Declare one to five distinct positive ridge strengths before fitting."
        )
    if (
        raw["validation_years"] != [2020, 2021, 2022]
        or raw["pca_dimensions"] != 64
        or type(raw["batch_size"]) is not int
        or raw["batch_size"] < 1
    ):
        raise ValueError(
            "Use the approved folds, 64 PCA dimensions, and a positive batch size."
        )
    output, batch_size = paths["output"], raw["batch_size"]
    output.mkdir(parents=True, exist_ok=True)
    resolved = {
        **raw,
        **{key: str(value) for key, value in paths.items()},
        "gru_runs": [str(p) for p in gru_runs],
        "split_configuration": yaml.safe_load(paths["split"].read_text()),
        "selection": (
            "Lowest mean aggregate signed-log RMSE over the three earlier temporal "
            "folds; ties use first declared strength"
        ),
        "summaries": (
            "Six/24-month signed-log means, observed-month occurrence fractions "
            "at abs(raw)>=1, observed fractions; final state/calendar/availability"
        ),
    }
    config_file = output / "config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != resolved:
        raise ValueError("Campaign configuration changed; use a new run directory.")
    if not config_file.exists():
        write_json(config_file, resolved)
    split = read_split(paths["split"])
    # Stop loading before reserved 2025, while keeping geography and split rules.
    split = replace(
        split,
        months=tuple(m for m in split.months if m < split.temporal["validation"][1]),
    )
    corpus = CellMonths.read(paths["dataset"], split)
    selection_scores: dict[str, list[list[float]]] = {
        name: [] for name in CONTROL_COLUMNS
    }
    for year in raw["validation_years"]:
        directory = output / str(year)
        directory.mkdir(exist_ok=True)
        fold = CellMonths(
            historical_split(split, year), corpus.values, corpus.available
        )
        preprocessing_path = directory / "preprocessing.npz"
        if preprocessing_path.exists():
            preprocessing = Preprocessing.read(preprocessing_path)
        else:
            preprocessing = Preprocessing.fit(fold)
            preprocessing.save(preprocessing_path)
        training = WindowDataset(fold, preprocessing, "train")
        validation = WindowDataset(fold, preprocessing, "validation")
        write_json(
            directory / "config.json",
            {
                "training_targets": [
                    date.isoformat() for date in fold.split.temporal["train"]
                ],
                "validation_targets": [
                    date.isoformat() for date in fold.split.temporal["validation"]
                ],
                "training_samples": len(training),
                "validation_samples": len(validation),
                "input_fit_start": fold.split.months[0].isoformat(),
                "input_fit_end_inclusive": fold.split.months[
                    training.starts[-1] - 1
                ].isoformat(),
                "ridge_strengths": strengths,
            },
        )
        fit_controls(
            training,
            directory,
            {name: strengths for name in CONTROL_COLUMNS},
            64,
            batch_size,
            device,
        )
        evaluate_controls(
            validation,
            directory,
            "temporal",
            {name: strengths for name in CONTROL_COLUMNS},
            batch_size,
            device,
        )
        evaluate_simple(validation, directory, "temporal", batch_size, device)
        for name in CONTROL_COLUMNS:
            selection_scores[name].append(
                [
                    json.loads(
                        prediction_path(directory, "temporal", f"{name}-{v:g}")
                        .with_suffix(".json")
                        .read_text()
                    )["scores"]["overall"]["signed_log_rmse"]
                    for v in strengths
                ]
            )
    selected = {
        name: strengths[int(np.asarray(scores).mean(0).argmin())]
        for name, scores in selection_scores.items()
    }
    write_json(
        output / "selection.json",
        {"strengths": strengths, "fold_rmse": selection_scores, "selected": selected},
    )
    directory = output / "original"
    directory.mkdir(exist_ok=True)
    preprocessing = Preprocessing.read(paths["preprocessing"])
    if not (directory / "preprocessing.npz").exists():
        preprocessing.save(directory / "preprocessing.npz")
    training = WindowDataset(corpus, preprocessing, "train")
    # Only the selected strength is fitted on the original training population.
    fit_controls(
        training,
        directory,
        {name: [strength] for name, strength in selected.items()},
        64,
        batch_size,
        device,
    )
    parent_resolution = resolved["split_configuration"]["geographic"][
        "parent_resolution"
    ]
    for group, geographic in (("temporal", False), ("geographic", True)):
        dataset = WindowDataset(
            corpus, preprocessing, "validation", geographic=geographic
        )
        parent = parent_resolution if geographic else None
        parent_ids = (
            tuple(resolved["split_configuration"]["geographic"]["validation_parents"])
            if geographic
            else ()
        )
        evaluate_controls(
            dataset,
            directory,
            group,
            {name: [strength] for name, strength in selected.items()},
            batch_size,
            device,
            parent,
            parent_ids,
        )
        evaluate_simple(
            dataset, directory, group, batch_size, device, parent, parent_ids
        )
        evaluate_references(
            dataset,
            directory,
            group,
            paths["tree_run"],
            gru_runs,
            batch_size,
            device,
            parent,
            parent_ids,
        )
    write_json(output / "building-additions.json", building_diagnostics(corpus))
    torch.cuda.synchronize()
    runtime_path = output / "runtime.json"
    if runtime_path.exists():
        runtime_path = output / "resume-runtime.json"
    write_json(
        runtime_path,
        {
            "seconds_this_invocation": time.monotonic() - started,
            "peak_process_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "gpu": torch.cuda.get_device_name(device),
        },
    )
    print(f"Campaign completed in {output}. Selected strengths: {selected}", flush=True)
