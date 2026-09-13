"""Aligned cell histories, training-only input scaling, and PyTorch windows."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from atlas.features import FEATURE_NAMES
from atlas.splits import Split

TARGET_NAMES = tuple(name for name in FEATURE_NAMES if not name.startswith("state_"))
TARGET_COLUMNS = [FEATURE_NAMES.index(name) for name in TARGET_NAMES]
INPUT_NAMES = (
    *FEATURE_NAMES,
    "calendar_sin",
    "calendar_cos",
    *(f"available_{name}" for name in FEATURE_NAMES),
)


@dataclass
class CellMonths:
    split: Split
    values: np.ndarray
    available: np.ndarray

    @classmethod
    def read(cls, directory: Path, split: Split) -> CellMonths:
        """Load ordered rows with structural checks, without summarizing targets."""
        metadata = json.loads((directory / "dataset.json").read_text())
        if (
            tuple(metadata["cells"]) != split.cells
            or tuple(metadata["feature_columns"]) != FEATURE_NAMES
        ):
            raise ValueError(
                "Corpus cells or features differ from the experiment contract."
            )
        values = np.zeros(
            (len(split.cells), len(split.months), len(FEATURE_NAMES)), dtype=np.float32
        )
        available = np.zeros_like(values, dtype=bool)
        for index, month in enumerate(split.months):
            table = pq.read_table(directory / f"{month:%Y-%m}.parquet").sort_by("cell")
            if tuple(table["cell"].to_pylist()) != split.cells or table[
                "month"
            ].to_pylist() != [month] * len(split.cells):
                raise ValueError(
                    "Corpus rows have missing/duplicate cells or incorrect months."
                )
            row_available = table["available"].to_numpy()
            for column, name in enumerate(FEATURE_NAMES):
                data = table[name].to_numpy()
                valid = np.isfinite(data)
                if not np.array_equal(valid, row_available):
                    raise ValueError("Feature nulls disagree with row availability.")
                available[:, index, column] = valid
                values[:, index, column] = np.where(valid, data, 0)
        return cls(split, values, available)


def signed_log(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


@dataclass(frozen=True)
class Preprocessing:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, corpus: CellMonths) -> Preprocessing:
        """Fit once per unique training input cell-month; exclude all holdouts."""
        split = corpus.split
        selected = set(split.eligible_cells("train"))
        cells = [i for i, cell in enumerate(split.cells) if cell in selected]
        months = sorted(
            {
                m
                for start in split.target_starts("train")
                for m in range(start - split.input_months, start)
            }
        )
        if not cells or not months:
            raise ValueError("Training has no eligible input windows.")
        total = np.zeros(len(FEATURE_NAMES))
        squares = np.zeros_like(total)
        count = np.zeros_like(total)
        for offset in range(0, len(cells), 128):
            indices = np.ix_(
                cells[offset : offset + 128], months, range(len(FEATURE_NAMES))
            )
            valid = corpus.available[indices]
            values = np.where(
                valid, signed_log(corpus.values[indices]).astype(np.float64), 0
            )
            total += values.sum(axis=(0, 1))
            squares += np.square(values).sum(axis=(0, 1))
            count += valid.sum(axis=(0, 1))
        if (count == 0).any():
            raise ValueError("A feature has no available training input observations.")
        mean = total / count
        scale = np.sqrt(np.maximum(squares / count - mean**2, 0))
        scale[scale < 1e-6] = 1
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def save(self, path: Path) -> None:
        with path.open("xb") as stream:
            np.savez(
                stream,
                mean=self.mean,
                scale=self.scale,
                features=np.array(FEATURE_NAMES),
                transform="signed_log1p",
            )

    @classmethod
    def read(cls, path: Path) -> Preprocessing:
        with np.load(path, allow_pickle=False) as data:
            if (
                tuple(data["features"]) != FEATURE_NAMES
                or str(data["transform"]) != "signed_log1p"
            ):
                raise ValueError("Preprocessing does not match the input features.")
            return cls(data["mean"], data["scale"])


class Sample(TypedDict):
    inputs: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    cell: str
    cutoff: str


class WindowDataset(Dataset[Sample]):
    """Enumerate every eligible cell/cutoff, retaining empty and unavailable rows."""

    def __init__(
        self,
        corpus: CellMonths,
        preprocessing: Preprocessing,
        partition: str,
        *,
        geographic: bool = False,
    ) -> None:
        self.corpus = corpus
        self.preprocessing = preprocessing
        self.starts = corpus.split.target_starts(partition)
        eligible = set(corpus.split.eligible_cells(partition, geographic=geographic))
        self.positions = [
            i for i, cell in enumerate(corpus.split.cells) if cell in eligible
        ]
        month = np.array([time.month - 1 for time in corpus.split.months])
        angle = 2 * np.pi * month / 12
        self.calendar = np.stack((np.sin(angle), np.cos(angle)), axis=-1).astype(
            np.float32
        )

    def __len__(self) -> int:
        return len(self.positions) * len(self.starts)

    def input_batch(self, indices: np.ndarray) -> np.ndarray:
        """Build the same permitted inputs for neural and tree models."""
        if indices.ndim != 1 or ((indices < 0) | (indices >= len(self))).any():
            raise IndexError("Input batch indices fall outside this partition.")
        cells = np.asarray(self.positions)[indices // len(self.starts)]
        starts = np.asarray(self.starts)[indices % len(self.starts)]
        months = starts[:, None] + np.arange(-self.corpus.split.input_months, 0)
        valid = self.corpus.available[cells[:, None], months]
        values = self.corpus.values[cells[:, None], months]
        normalized = (
            signed_log(values) - self.preprocessing.mean
        ) / self.preprocessing.scale
        return np.concatenate(
            (np.where(valid, normalized, 0), self.calendar[months], valid), axis=-1
        ).astype(np.float32)

    def target_batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return raw targets and masks in exactly the input batch's order."""
        if indices.ndim != 1 or ((indices < 0) | (indices >= len(self))).any():
            raise IndexError("Target batch indices fall outside this partition.")
        cells = np.asarray(self.positions)[indices // len(self.starts)]
        starts = np.asarray(self.starts)[indices % len(self.starts)]
        months = starts[:, None] + np.arange(self.corpus.split.target_months)
        return (
            self.corpus.values[cells[:, None], months][:, :, TARGET_COLUMNS],
            self.corpus.available[cells[:, None], months][:, :, TARGET_COLUMNS],
        )

    def __getitem__(self, index: int) -> Sample:
        if not 0 <= index < len(self):
            raise IndexError(index)
        cell = self.positions[index // len(self.starts)]
        start = self.starts[index % len(self.starts)]
        split = self.corpus.split
        target_slice = slice(start, start + split.target_months)
        inputs = self.input_batch(np.array([index]))[0]
        targets = self.corpus.values[cell, target_slice][:, TARGET_COLUMNS].copy()
        target_mask = self.corpus.available[cell, target_slice][
            :, TARGET_COLUMNS
        ].copy()
        return {
            "inputs": torch.from_numpy(inputs),
            "targets": torch.from_numpy(targets),
            "target_mask": torch.from_numpy(target_mask),
            "cell": split.cells[cell],
            "cutoff": split.months[start - 1].date().isoformat(),
        }
