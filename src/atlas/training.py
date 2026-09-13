"""Direct GPU training, epoch checkpoints, and validation of the neural models."""

import json
import math
import resource
import time
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from atlas.baselines import target_column, write_json
from atlas.linear import control_inputs
from atlas.metrics import Scores, score_column, transform
from atlas.model import (
    ResidualMLP,
    TemporalGRU,
    family_mean,
    masked_loss,
    project_counts,
)
from atlas.samples import (
    INPUT_NAMES,
    TARGET_NAMES,
    CellMonths,
    Preprocessing,
    WindowDataset,
)
from atlas.splits import read_split


@dataclass(frozen=True)
class Training:
    hidden_size: int
    layers: int
    batch_size: int
    epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    seed: int

    def __post_init__(self) -> None:
        if (
            self.hidden_size not in range(64, 129)
            or self.layers not in (1, 2)
            or min(self.batch_size, self.epochs) < 1
            or self.patience < 0
            or self.seed < 0
            or not all(
                math.isfinite(v)
                for v in (self.learning_rate, self.weight_decay, self.gradient_clip)
            )
            or min(self.learning_rate, self.gradient_clip) <= 0
            or self.weight_decay < 0
        ):
            raise ValueError("Invalid neural training settings.")


def gpu_device() -> torch.device:
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("Neural execution requires the AMD GPU inside Compose.")
    torch.set_num_threads(6)
    return torch.device("cuda:0")


def save_checkpoint(
    path: Path,
    model: TemporalGRU | ResidualMLP,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    step: int,
    best_rmse: float,
    bad_epochs: int,
    configuration: dict[str, object],
) -> None:
    """Replace only a complete checkpoint; interrupted writes leave its predecessor."""
    device = next(model.parameters()).device
    temporary = path.with_suffix(".partial")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_rmse": best_rmse,
            "bad_epochs": bad_epochs,
            "configuration": configuration,
            "random_state": torch.get_rng_state(),
            "accelerator_random_state": (
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            ),
        },
        temporary,
    )
    temporary.replace(path)


def restore_checkpoint(
    path: Path,
    model: TemporalGRU | ResidualMLP,
    optimizer: torch.optim.Optimizer,
    configuration: dict[str, object],
) -> tuple[int, int, float, int]:
    device = next(model.parameters()).device
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved["configuration"] != configuration:
        raise ValueError("Checkpoint configuration differs from this run.")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["random_state"].cpu())
    if saved["accelerator_random_state"] is not None:
        torch.cuda.set_rng_state(saved["accelerator_random_state"].cpu(), device)
    return saved["epoch"], saved["step"], saved["best_rmse"], saved["bad_epochs"]


def batch(
    dataset: WindowDataset,
    indices: np.ndarray,
    device: torch.device,
    model: TemporalGRU | ResidualMLP | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets, mask = dataset.target_batch(indices)
    return (
        torch.as_tensor(dataset.input_batch(indices), device=device)
        if model is None
        else model_inputs(model, dataset, indices, device),
        torch.as_tensor(targets, device=device),
        torch.as_tensor(mask, device=device),
    )


def model_inputs(
    model: TemporalGRU | ResidualMLP,
    dataset: WindowDataset,
    indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(model, ResidualMLP):
        return control_inputs(dataset, indices, device)[:, model.input_columns].float()
    return torch.as_tensor(dataset.input_batch(indices), device=device)


@dataclass
class CachedWindows:
    inputs: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor

    def batch(
        self, indices: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = torch.as_tensor(indices, device=self.inputs.device)
        return self.inputs[rows], self.targets[rows], self.mask[rows]


def cache_windows(
    model: ResidualMLP, dataset: WindowDataset, device: torch.device
) -> CachedWindows:
    """Prepare fixed inputs once for the repeated dense-network epochs on GPU."""
    inputs = torch.empty((len(dataset), model.input_size), device=device)
    targets = torch.empty((len(dataset), 6, len(TARGET_NAMES)), device=device)
    mask = torch.empty_like(targets, dtype=torch.bool)
    for offset in range(0, len(dataset), 4096):
        rows = np.arange(offset, min(offset + 4096, len(dataset)))
        inputs[offset : offset + len(rows)] = model_inputs(model, dataset, rows, device)
        raw, available = dataset.target_batch(rows)
        targets[offset : offset + len(rows)] = torch.as_tensor(raw, device=device)
        mask[offset : offset + len(rows)] = torch.as_tensor(available, device=device)
    return CachedWindows(inputs, targets, mask)


def scheduled_rate(
    initial: float, epoch: int, milestones: list[int], factor: float
) -> float:
    """Use completed epoch count so checkpoint resume follows the same schedule."""
    return initial * factor ** sum(epoch >= milestone for milestone in milestones)


@torch.inference_mode()
def validation_rmse(
    model: TemporalGRU | ResidualMLP,
    dataset: WindowDataset,
    batch_size: int,
    device: torch.device,
    cached: CachedWindows | None = None,
) -> float:
    """Select checkpoints with magnitude error on all temporal-validation rows."""
    model.eval()
    total = torch.zeros((6, len(TARGET_NAMES)), dtype=torch.float64, device=device)
    counts = torch.zeros_like(total)
    for offset in range(0, len(dataset), batch_size):
        indices = np.arange(offset, min(offset + batch_size, len(dataset)))
        inputs, targets, mask = (
            cached.batch(indices)
            if cached is not None
            else batch(dataset, indices, device, model)
        )
        prediction, _ = model(inputs)
        if not torch.isfinite(prediction).all():
            raise ValueError("Non-finite neural validation predictions.")
        errors = project_counts(prediction).double() - transform(targets.double())
        total += torch.where(mask, errors, 0).square().sum(dim=0)
        counts += mask.sum(dim=0)
    if not counts.any():
        raise ValueError("Validation has no observed targets.")
    return float(family_mean(total / counts.clamp_min(1), counts > 0).sqrt())


def sample_keys(dataset: WindowDataset, indices: np.ndarray) -> dict[str, list[str]]:
    return {
        "cell": [
            dataset.corpus.split.cells[dataset.positions[i // len(dataset.starts)]]
            for i in indices
        ],
        "cutoff": [
            dataset.corpus.split.months[dataset.starts[i % len(dataset.starts)] - 1]
            .date()
            .isoformat()
            for i in indices
        ],
    }


def score_predictions(
    dataset: WindowDataset, predictions: torch.Tensor
) -> dict[str, object]:
    """Apply the existing scorer to aligned, transformed multi-output forecasts."""
    if predictions.shape != (len(dataset), 6, len(TARGET_NAMES)):
        raise ValueError("Predictions do not match this evaluation dataset.")
    scores = Scores()
    for h in range(6):
        for c, name in enumerate(TARGET_NAMES):
            target, mask = (
                torch.as_tensor(v, device=predictions.device)
                for v in target_column(dataset, h, c)
            )
            scores.record(
                h,
                c,
                score_column(
                    target,
                    predictions[:, h, c],
                    mask,
                    count_target=not name.startswith("net_"),
                ),
            )
    return {"samples": len(dataset), **scores.summary()}


@torch.inference_mode()
def evaluate_model(
    model: TemporalGRU | ResidualMLP,
    dataset: WindowDataset,
    batch_size: int,
    device: torch.device,
    destination: Path,
    cached: CachedWindows | None = None,
) -> dict[str, object]:
    """Export one embedding and six-month forecast per cell/cutoff, then score."""
    model.eval()
    predictions = torch.empty((len(dataset), 6, len(TARGET_NAMES)), device=device)
    temporary = destination.with_suffix(".partial")
    schema = pa.schema(
        [
            ("cell", pa.string()),
            ("cutoff", pa.string()),
            ("embedding", pa.list_(pa.float32(), model.embedding_size)),
            ("prediction_log", pa.list_(pa.float32(), 6 * len(TARGET_NAMES))),
        ],
        metadata={
            "target_names": json.dumps(TARGET_NAMES),
            "forecast_order": "horizon-major; months 1 through 6",
            "prediction_units": "signed_log1p; count forecasts projected to >= 0",
            "decode": "sign(z) * expm1(abs(z))",
        },
    )
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for offset in range(0, len(dataset), batch_size):
            indices = np.arange(offset, min(offset + batch_size, len(dataset)))
            inputs = (
                cached.inputs[offset : offset + len(indices)]
                if cached is not None
                else model_inputs(model, dataset, indices, device)
            )
            prediction, embedding = model(inputs)
            prediction = project_counts(prediction)
            if (
                not torch.isfinite(prediction).all()
                or not torch.isfinite(embedding).all()
            ):
                raise ValueError("Non-finite neural forecasts or embeddings.")
            predictions[offset : offset + len(indices)] = prediction
            keys = sample_keys(dataset, indices)
            writer.write_table(
                pa.Table.from_arrays(
                    [
                        pa.array(keys["cell"]),
                        pa.array(keys["cutoff"]),
                        pa.FixedSizeListArray.from_arrays(
                            embedding.cpu().numpy().ravel(), model.embedding_size
                        ),
                        pa.FixedSizeListArray.from_arrays(
                            prediction.cpu().numpy().ravel(), 6 * len(TARGET_NAMES)
                        ),
                    ],
                    schema=schema,
                )
            )
    scores = score_predictions(dataset, predictions)
    temporary.replace(destination)
    return scores


def train(config_path: Path, resume: Path | None = None) -> None:
    """Train on permitted histories and evaluate the best epoch on validation only."""
    run_started = time.monotonic()
    device = gpu_device()
    raw = tomllib.loads(config_path.read_text())
    settings = Training(**raw["training"])
    model_settings = raw.get("model")
    schedule = raw.get("schedule", {"milestones": [], "factor": 1.0})
    milestones, factor = schedule["milestones"], schedule["factor"]
    if (
        not isinstance(milestones, list)
        or any(type(e) is not int or not 0 < e < settings.epochs for e in milestones)
        or milestones != sorted(set(milestones))
        or not math.isfinite(factor)
        or not 0 < factor <= 1
    ):
        raise ValueError("Invalid epoch-based learning-rate schedule.")
    paths = {
        key: (config_path.parent / raw[key]).resolve()
        for key in ("dataset", "split", "preprocessing", "output")
    }
    split = read_split(paths["split"])
    split = replace(
        split,
        months=tuple(m for m in split.months if m < split.temporal["validation"][1]),
    )
    if (split.input_months, split.target_months) != (24, 6):
        raise ValueError("The current learner requires the frozen 24/6-month windows.")
    preprocessing = Preprocessing.read(paths["preprocessing"])
    configuration = {
        **{key: str(path) for key, path in paths.items()},
        "training": asdict(settings),
        "input_names": list(INPUT_NAMES),
        "target_names": list(TARGET_NAMES),
        "split_configuration": yaml.safe_load(paths["split"].read_text()),
        "input_mean": preprocessing.mean.tolist(),
        "input_scale": preprocessing.scale.tolist(),
        "checkpoint_selection": "minimum full temporal-validation signed-log RMSE",
    }
    if model_settings is not None:
        configuration["model"] = model_settings
    if "schedule" in raw:
        configuration["schedule"] = schedule
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    config_file = output / "config.json"
    if config_file.exists():
        if json.loads(config_file.read_text()) != configuration:
            raise ValueError("Run configuration changed. Use a new output directory.")
        if resume is None and (output / "latest.pt").exists():
            raise ValueError("This run has a checkpoint. Use --resume with latest.pt.")
    elif resume is not None:
        raise ValueError(
            "Resume requires the original run directory and configuration."
        )
    else:
        write_json(config_file, configuration)
    corpus = CellMonths.read(paths["dataset"], split)
    training = WindowDataset(corpus, preprocessing, "train")
    validation = {
        "temporal": WindowDataset(corpus, preprocessing, "validation"),
        "geographic": WindowDataset(
            corpus, preprocessing, "validation", geographic=True
        ),
    }
    if not len(training) or any(not len(v) for v in validation.values()):
        raise ValueError("Training and both validation groups must contain samples.")
    torch.manual_seed(settings.seed)
    model = (
        TemporalGRU(settings.hidden_size, settings.layers)
        if model_settings is None
        else ResidualMLP(**model_settings)
    ).to(device)
    cached: dict[str, CachedWindows] = {}
    if isinstance(model, ResidualMLP):
        torch.cuda.reset_peak_memory_stats(device)
        required = (len(training) + sum(len(v) for v in validation.values())) * (
            4 * model.input_size + 5 * 222
        )
        free, _ = torch.cuda.mem_get_info(device)
        if required + 2 * 1024**3 > free:
            raise ValueError(
                "This MLP comparison needs room for its GPU inputs "
                "plus 2 GiB of working memory."
            )
        cached["train"] = cache_windows(model, training, device)
        model.fit_input_scaling(cached["train"].inputs)
        for name, dataset in validation.items():
            cached[name] = cache_windows(model, dataset, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    epoch, step, best_rmse, bad_epochs = 0, 0, math.inf, 0
    if resume is not None:
        if resume.resolve() != (output / "latest.pt").resolve():
            raise ValueError(
                "Resume from this run's latest.pt to retain early stopping."
            )
        epoch, step, best_rmse, bad_epochs = restore_checkpoint(
            resume, model, optimizer, configuration
        )
        if not (output / "best.pt").exists():
            raise ValueError("The best checkpoint is missing from the resumed run.")
    print(
        f"{type(model).__name__} GPU: {torch.cuda.get_device_name(device)}; "
        f"{sum(p.numel() for p in model.parameters()):,} parameters; "
        f"{len(training):,} training samples; starting after epoch {epoch}",
        flush=True,
    )
    while epoch < settings.epochs and (
        settings.patience == 0 or bad_epochs < settings.patience
    ):
        started = time.monotonic()
        learning_rate = scheduled_rate(
            settings.learning_rate, epoch, milestones, factor
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        indices = np.random.default_rng(settings.seed + epoch).permutation(
            len(training)
        )
        total_loss, observed_batches = 0.0, 0
        for offset in range(0, len(indices), settings.batch_size):
            rows = indices[offset : offset + settings.batch_size]
            inputs, targets, mask = (
                cached["train"].batch(rows) if cached else batch(training, rows, device)
            )
            if not mask.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(inputs)
            loss = masked_loss(prediction, targets, mask)
            if not torch.isfinite(loss):
                raise ValueError(
                    "Non-finite neural training loss; last checkpoint retained."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            total_loss += float(loss.detach())
            observed_batches += 1
            step += 1
        if not observed_batches:
            raise ValueError("Training has no observed targets.")
        rmse = validation_rmse(
            model,
            validation["temporal"],
            settings.batch_size,
            device,
            cached.get("temporal"),
        )
        epoch += 1
        improved = rmse < best_rmse
        best_rmse = min(best_rmse, rmse)
        bad_epochs = 0 if improved else bad_epochs + 1
        for name in ["best", "latest"] if improved else ["latest"]:
            save_checkpoint(
                output / f"{name}.pt",
                model,
                optimizer,
                epoch=epoch,
                step=step,
                best_rmse=best_rmse,
                bad_epochs=bad_epochs,
                configuration=configuration,
            )
        record = {
            "epoch": epoch,
            "step": step,
            "mean_batch_training_loss": total_loss / observed_batches,
            "temporal_validation_rmse": rmse,
            "best_rmse": best_rmse,
            "seconds": time.monotonic() - started,
            "learning_rate": learning_rate,
        }
        with (output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(
            f"Epoch {epoch}/{settings.epochs}: training loss "
            f"{record['mean_batch_training_loss']:.6f}; validation RMSE {rmse:.6f}; "
            f"best {best_rmse:.6f}; lr {learning_rate:g}; {record['seconds']:.1f} s",
            flush=True,
        )
    best = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model"])
    metrics: dict[str, object] = {"best_epoch": best["epoch"], "training_epochs": epoch}
    for name, dataset in validation.items():
        scores = evaluate_model(
            model,
            dataset,
            settings.batch_size,
            device,
            output / f"{name}.parquet",
            cached.get(name),
        )
        metrics[name] = scores
        write_json(output / "metrics.json", metrics)
        print(
            f"{name}: RMSE {scores['signed_log_rmse']:.6f}; "
            f"AP {scores['average_precision']:.6f}; {len(dataset):,} exports",
            flush=True,
        )
    if isinstance(model, ResidualMLP) and not (output / "runtime.json").exists():
        write_json(
            output / "runtime.json",
            {
                "seconds_this_invocation": time.monotonic() - run_started,
                "parameters": sum(p.numel() for p in model.parameters()),
                "peak_process_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024,
                "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device)
                / 2**20,
                "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            },
        )
