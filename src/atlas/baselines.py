"""Run the three required baselines on training and validation only."""

import json
import os
import tempfile
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
import yaml

# Both libraries must use Torch's working WSL ROCm runtime.
import lightgbm as lgb  # isort: skip

from atlas.metrics import Scores, score_column, transform
from atlas.samples import (
    INPUT_NAMES,
    TARGET_COLUMNS,
    TARGET_NAMES,
    CellMonths,
    Preprocessing,
    WindowDataset,
    signed_log,
)
from atlas.splits import read_split
from atlas.trees import InputSequence, RegressionTrees


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def target_column(
    dataset: WindowDataset, horizon: int, column: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read one target without changing the dataset's cell/cutoff order."""
    cells = np.repeat(dataset.positions, len(dataset.starts))
    months = np.tile(dataset.starts, len(dataset.positions)) + horizon
    feature = TARGET_COLUMNS[column]
    return (
        dataset.corpus.values[cells, months, feature],
        dataset.corpus.available[cells, months, feature],
    )


def recent_rate(
    dataset: WindowDataset, months: int, device: torch.device
) -> torch.Tensor:
    """Mean raw monthly change over available recent inputs, repeated six times.

    A feature with no observed lookback values predicts zero. No target values
    enter this calculation. Signed net changes can cancel within the lookback.
    """
    if not 1 <= months <= dataset.corpus.split.input_months:
        raise ValueError("Recent rate must use between 1 and 24 input months.")
    result = torch.empty((len(dataset), len(TARGET_NAMES)), device=device)
    for offset in range(0, len(dataset), 4096):
        indices = np.arange(offset, min(offset + 4096, len(dataset)))
        cells = np.asarray(dataset.positions)[indices // len(dataset.starts)]
        starts = np.asarray(dataset.starts)[indices % len(dataset.starts)]
        times = starts[:, None] + np.arange(-months, 0)
        values = torch.as_tensor(
            dataset.corpus.values[cells[:, None], times][:, :, TARGET_COLUMNS],
            device=device,
        )
        mask = torch.as_tensor(
            dataset.corpus.available[cells[:, None], times][:, :, TARGET_COLUMNS],
            device=device,
        )
        rate = torch.where(mask, values, 0).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        result[offset : offset + len(indices)] = transform(rate)
    return result


def gpu_inputs(dataset: WindowDataset, device: torch.device) -> torch.Tensor:
    result = torch.empty((len(dataset), 24 * len(INPUT_NAMES)), device=device)
    for offset in range(0, len(dataset), 4096):
        indices = np.arange(offset, min(offset + 4096, len(dataset)))
        result[offset : offset + len(indices)] = torch.as_tensor(
            dataset.input_batch(indices).reshape(len(indices), -1), device=device
        )
    return result


def binned_dataset(
    dataset: WindowDataset,
    path: Path,
    params: dict[str, int | float | str | bool],
    reference: lgb.Dataset | None = None,
) -> lgb.Dataset:
    if reference is not None:
        # LightGBM 4.7's Sequence/binary reference loaders omit GPU metadata and
        # crash when attaching validation metrics. Its matrix loader initializes
        # that metadata correctly. Only validation needs this dense host buffer.
        values = np.empty((len(dataset), 24 * len(INPUT_NAMES)), dtype=np.float32)
        for offset in range(0, len(dataset), 4096):
            indices = np.arange(offset, min(offset + 4096, len(dataset)))
            values[offset : offset + len(indices)] = dataset.input_batch(
                indices
            ).reshape(len(indices), -1)
        return lgb.Dataset(
            values, label=np.zeros(len(dataset)), params=params, reference=reference
        ).construct()
    if path.exists():
        return lgb.Dataset(str(path), params=params, reference=reference).construct()
    result = lgb.Dataset(
        InputSequence(dataset),
        label=np.zeros(len(dataset)),
        params=params,
        reference=reference,
    ).construct()
    # LightGBM refuses to overwrite even a partial binary left by interruption.
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / "data.bin"
        result.save_binary(str(temporary))
        temporary.replace(path)
    return result


def train_baselines(config_path: Path) -> None:
    """Fit independent target/horizon trees; retain completed models on restart."""
    if os.environ.get("HIP_LAUNCH_BLOCKING") != "1":
        raise RuntimeError(
            "Use atlas train-baselines to enable synchronous HIP launches."
        )
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("Baseline execution requires the AMD GPU inside Compose.")
    device = torch.device("cuda:0")
    print(f"Baseline GPU: {torch.cuda.get_device_name(device)}", flush=True)
    raw = tomllib.loads(config_path.read_text())
    paths = {
        key: (config_path.parent / raw[key]).resolve()
        for key in ("dataset", "split", "preprocessing", "output")
    }
    recent_months, rounds, patience, seed = (
        int(raw[key])
        for key in ("recent_months", "rounds", "early_stopping_rounds", "seed")
    )
    if not 1 <= recent_months <= 24 or not 0 < patience < rounds:
        raise ValueError(
            "Invalid rate window, boosting rounds or early stopping patience."
        )
    tree_options = raw["trees"]
    if set(tree_options) != {
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_data_in_leaf",
        "feature_fraction",
        "lambda_l2",
    }:
        raise ValueError(
            "Provide the six declared tree settings from configs/baselines.toml."
        )
    params: dict[str, int | float | str | bool] = {
        **tree_options,
        "objective": "regression_l2",
        "metric": "l2",
        "device_type": "cuda",
        "num_gpu": 1,
        "gpu_device_id": 0,
        "num_threads": 6,
        "max_bin": 63,
        "bin_construct_sample_cnt": 20000,
        "feature_pre_filter": False,
        "zero_as_missing": False,
        "seed": seed,
        "data_random_seed": seed,
        "verbosity": -1,
    }
    resolved = {
        **{key: str(path) for key, path in paths.items()},
        "recent_months": recent_months,
        "recent_rate": "arithmetic mean of available raw inputs; empty lookback = zero",
        "rounds": rounds,
        "early_stopping_rounds": patience,
        "early_stopping": "per-target/horizon temporal-validation transformed MSE",
        "parameters": params,
        "input_names": INPUT_NAMES,
        "target_names": TARGET_NAMES,
        "input_months": 24,
        "target_months": 6,
        "lightgbm_version": lgb.__version__,
        "hip_launch_blocking": True,
        "split_configuration": yaml.safe_load(paths["split"].read_text()),
    }
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    config_file = output / "config.json"
    if config_file.exists():
        if json.loads(config_file.read_text()) != json.loads(json.dumps(resolved)):
            raise ValueError("Run configuration changed. Use a new output directory.")
    else:
        write_json(config_file, resolved)
    split = read_split(paths["split"])
    if (split.input_months, split.target_months) != (24, 6):
        raise ValueError("Baselines require the frozen 24/6-month windows.")
    corpus = CellMonths.read(paths["dataset"], split)
    preprocessing = Preprocessing.read(paths["preprocessing"])
    training = WindowDataset(corpus, preprocessing, "train")
    validation = {
        "temporal": WindowDataset(corpus, preprocessing, "validation"),
        "geographic": WindowDataset(
            corpus, preprocessing, "validation", geographic=True
        ),
    }
    metrics: dict[str, dict[str, object]] = {}
    for name, dataset in validation.items():
        tables = {"zero": Scores(), "recent_rate": Scores()}
        rate = recent_rate(dataset, recent_months, device)
        zero = torch.zeros(len(dataset), device=device)
        for h in range(6):
            for c, target_name in enumerate(TARGET_NAMES):
                target, available = (
                    torch.as_tensor(v, device=device)
                    for v in target_column(dataset, h, c)
                )
                for model_name, prediction in (
                    ("zero", zero),
                    ("recent_rate", rate[:, c]),
                ):
                    tables[model_name].record(
                        h,
                        c,
                        score_column(
                            target,
                            prediction,
                            available,
                            count_target=not target_name.startswith("net_"),
                        ),
                    )
        metrics[name] = {
            "samples": len(dataset),
            **{k: table.summary() for k, table in tables.items()},
        }
        del rate
    write_json(output / "metrics.json", metrics)
    print_table(metrics)

    print(f"Binning {len(training):,} training inputs in batches.", flush=True)
    train_data = binned_dataset(training, output / "train.bin", params)
    valid_data = binned_dataset(
        validation["temporal"], output / "validation.bin", params, train_data
    )
    inputs = {name: gpu_inputs(dataset, device) for name, dataset in validation.items()}
    tables = {name: Scores() for name in validation}
    models = output / "models"
    models.mkdir(exist_ok=True)
    for h in range(6):
        for c, target_name in enumerate(TARGET_NAMES):
            started = time.monotonic()
            model_path = models / f"h{h + 1}-{target_name}.txt"
            if not model_path.exists():
                for data, dataset in (
                    (train_data, training),
                    (valid_data, validation["temporal"]),
                ):
                    target, available = target_column(dataset, h, c)
                    if not available.any():
                        raise ValueError(
                            f"No observed {target_name} targets for this model."
                        )
                    data.set_label(signed_log(target))
                    data.set_weight(available.astype(np.float32))
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=rounds,
                    valid_sets=[valid_data],
                    callbacks=[lgb.early_stopping(patience, verbose=False)],
                )
                temporary = model_path.with_suffix(".partial")
                model.save_model(str(temporary))
                temporary.replace(model_path)
                del model
            forest = RegressionTrees(model_path.read_text()).to(device)
            with torch.inference_mode():
                for name, dataset in validation.items():
                    prediction = torch.empty(
                        len(dataset), dtype=torch.float64, device=device
                    )
                    for offset in range(0, len(dataset), 2048):
                        prediction[offset : offset + 2048] = forest(
                            inputs[name][offset : offset + 2048]
                        )
                    target, available = (
                        torch.as_tensor(v, device=device)
                        for v in target_column(dataset, h, c)
                    )
                    tables[name].record(
                        h,
                        c,
                        score_column(
                            target,
                            prediction,
                            available,
                            count_target=not target_name.startswith("net_"),
                        ),
                    )
                    metrics[name]["boosted_trees"] = {
                        "completed_models": h * 37 + c + 1,
                        "expected_models": 222,
                        **tables[name].summary(),
                    }
            write_json(output / "metrics.json", metrics)
            print(
                f"Completed tree {h * 37 + c + 1}/222: h{h + 1} {target_name}; "
                f"{time.monotonic() - started:.1f} s",
                flush=True,
            )
            del forest
    print_table(metrics)


def print_table(metrics: dict[str, dict[str, object]]) -> None:
    print("Validation  Baseline         Signed-log RMSE  Average precision", flush=True)
    for partition, models in metrics.items():
        for name, scores in models.items():
            if isinstance(scores, dict):
                values = [
                    f"{scores[key]:.6f}" if scores[key] is not None else "NA"
                    for key in ("signed_log_rmse", "average_precision")
                ]
                print(
                    f"{partition:11} {name:16} {values[0]:>15} {values[1]:>18}",
                    flush=True,
                )
