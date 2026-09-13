"""Batched full-data ridge and PCA controls for the development campaign."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime

import numpy as np
import torch

from atlas.baselines import target_column
from atlas.features import FEATURE_NAMES
from atlas.metrics import transform
from atlas.samples import INPUT_NAMES, TARGET_COLUMNS, WindowDataset
from atlas.splits import Split

ORDERED_SIZE = 24 * len(INPUT_NAMES)
SUMMARY_SIZE = 2 * 3 * len(TARGET_COLUMNS)
JOINT_SIZE = ORDERED_SIZE + SUMMARY_SIZE
STATE_COLUMNS = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith("state_")]
FINAL_COLUMNS = [
    23 * len(INPUT_NAMES) + i
    for i in (*STATE_COLUMNS, 51, 52, *(53 + c for c in STATE_COLUMNS))
]
CONTROL_COLUMNS = {
    "ordered": list(range(ORDERED_SIZE)),
    "pca": list(range(ORDERED_SIZE)),
    "state": FINAL_COLUMNS,
    "summary": FINAL_COLUMNS + list(range(ORDERED_SIZE, JOINT_SIZE)),
}


def historical_split(split: Split, validation_year: int) -> Split:
    """Change only development dates; retain every geographic exclusion."""
    start = datetime(validation_year, 1, 1, tzinfo=UTC)
    end = datetime(validation_year + 1, 1, 1, tzinfo=UTC)
    if not split.temporal["train"][0] < start < end <= split.temporal["train"][1]:
        raise ValueError(
            "Historical validation must be inside original training dates."
        )
    return replace(
        split,
        temporal={
            **split.temporal,
            "train": (split.temporal["train"][0], start),
            "validation": (start, end),
        },
    )


def change_history(
    dataset: WindowDataset, indices: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    cells = np.asarray(dataset.positions)[indices // len(dataset.starts)]
    starts = np.asarray(dataset.starts)[indices % len(dataset.starts)]
    months = starts[:, None] + np.arange(-24, 0)
    return (
        torch.as_tensor(
            dataset.corpus.values[cells[:, None], months][:, :, TARGET_COLUMNS],
            device=device,
            dtype=torch.float64,
        ),
        torch.as_tensor(
            dataset.corpus.available[cells[:, None], months][:, :, TARGET_COLUMNS],
            device=device,
        ),
    )


def history_summaries(values: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    """Six/24-month log means, occurrence fractions, and observed fractions.

    Fractions of active months use observed months as the denominator. Empty
    lookbacks produce zero and a zero observed fraction, preserving missingness.
    """
    parts = []
    for months in (6, 24):
        raw, mask = values[:, -months:], available[:, -months:]
        count = mask.sum(1).to(raw.dtype)
        parts.extend(
            (
                torch.where(mask, transform(raw), 0).sum(1) / count.clamp_min(1),
                ((raw.abs() >= 1) & mask).sum(1) / count.clamp_min(1),
                count / months,
            )
        )
    return torch.cat(parts, dim=1)


def control_inputs(
    dataset: WindowDataset, indices: np.ndarray, device: torch.device
) -> torch.Tensor:
    ordered = torch.as_tensor(dataset.input_batch(indices), device=device).double()
    values, mask = change_history(dataset, indices, device)
    return torch.cat((ordered.flatten(1), history_summaries(values, mask)), dim=1)


@dataclass
class Moments:
    """Uncentered sums in float64; memory depends on columns, not sample count."""

    count: int
    total: torch.Tensor
    gram: torch.Tensor
    target_total: torch.Tensor
    cross: torch.Tensor

    @classmethod
    def empty(cls, columns: int, outputs: int, device: torch.device) -> Moments:
        def zeros(*shape: int) -> torch.Tensor:
            return torch.zeros(shape, dtype=torch.float64, device=device)

        return cls(
            0,
            zeros(columns),
            zeros(columns, columns),
            zeros(outputs),
            zeros(columns, outputs),
        )

    def add(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        x, y = inputs.double(), targets.double()
        self.count += len(x)
        self.total += x.sum(0)
        self.gram += x.T @ x
        self.target_total += y.sum(0)
        self.cross += x.T @ y

    def covariance(self) -> torch.Tensor:
        mean = self.total / self.count
        result = self.gram / self.count - mean[:, None] * mean[None, :]
        return (result + result.T) / 2


def accumulate(
    dataset: WindowDataset,
    device: torch.device,
    batch_size: int,
    indices: np.ndarray | None = None,
) -> Moments:
    result = Moments.empty(JOINT_SIZE, 6 * len(TARGET_COLUMNS), device)
    selected = np.arange(len(dataset)) if indices is None else indices
    for offset in range(0, len(selected), batch_size):
        rows = selected[offset : offset + batch_size]
        raw, mask = dataset.target_batch(rows)
        target = torch.as_tensor(np.where(mask, raw, 0), device=device).double()
        result.add(control_inputs(dataset, rows, device), transform(target).flatten(1))
    if result.count == 0 or not torch.isfinite(result.gram).all():
        raise ValueError("No training rows or non-finite training moments.")
    return result


def target_groups(dataset: WindowDataset) -> list[tuple[list[int], np.ndarray]]:
    """Share a solve only between outputs with exactly the same observed rows."""
    groups: dict[bytes, tuple[list[int], np.ndarray]] = {}
    for h in range(6):
        for c in range(len(TARGET_COLUMNS)):
            _, mask = target_column(dataset, h, c)
            key = np.packbits(mask).tobytes()
            if key not in groups:
                groups[key] = ([], np.flatnonzero(mask))
            groups[key][0].append(h * len(TARGET_COLUMNS) + c)
    return list(groups.values())


def representation_maps(
    moments: Moments, pca_dimensions: int
) -> dict[str, torch.Tensor]:
    """Fit each representation's scaling on all training rows, without labels."""
    covariance = moments.covariance()
    device = covariance.device
    maps = {}
    for name, columns in CONTROL_COLUMNS.items():
        selected = covariance[columns][:, columns]
        if name == "pca":
            _, axes = torch.linalg.eigh(selected)
            basis = axes[:, -pca_dimensions:].flip(1)
            variance = (basis * (selected @ basis)).sum(0)
        else:
            basis = torch.eye(len(columns), dtype=torch.float64, device=device)
            variance = selected.diag()
        scale = variance.clamp_min(0).sqrt()
        scale = torch.where(scale < 1e-6, 1, scale)
        projection = covariance.new_zeros((len(covariance), basis.shape[1]))
        projection[columns] = basis / scale
        maps[name] = projection
    return maps


def ridge_solutions(
    moments: Moments, projection: torch.Tensor, strengths: list[float]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Solve centered ridge, then fuse the input map and unpenalized intercept.

    Covariances divide by observed sample count, so strengths have the same
    meaning in each fold and for outputs with different availability.
    """
    mean = moments.total / moments.count
    target_mean = moments.target_total / moments.count
    covariance = projection.T @ moments.covariance() @ projection
    cross = projection.T @ (
        moments.cross / moments.count - mean[:, None] * target_mean[None, :]
    )
    eigenvalues, axes = torch.linalg.eigh((covariance + covariance.T) / 2)
    rotated = axes.T @ cross
    solutions = []
    for strength in strengths:
        if not np.isfinite(strength) or strength <= 0:
            raise ValueError("Ridge strengths must be finite and positive.")
        coefficient = axes @ (rotated / (eigenvalues[:, None] + strength))
        if not torch.isfinite(coefficient).all():
            raise ValueError("Non-finite ridge coefficients.")
        weights = projection @ coefficient
        solutions.append((weights, target_mean - mean @ weights))
    return solutions
