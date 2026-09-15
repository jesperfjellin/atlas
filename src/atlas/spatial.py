"""Observed one-ring summaries with the experiment's geographic exclusions."""

from dataclasses import dataclass

import h3
import numpy as np
import torch

from atlas.features import FEATURE_NAMES
from atlas.linear import STATE_COLUMNS
from atlas.metrics import transform
from atlas.samples import TARGET_COLUMNS, TARGET_NAMES, WindowDataset
from atlas.splits import Split

NEIGHBOUR_NAMES = (
    *(f"neighbour_mean_log_{FEATURE_NAMES[c]}" for c in STATE_COLUMNS),
    *(f"neighbour_observed_{FEATURE_NAMES[c]}" for c in STATE_COLUMNS),
    *(
        f"neighbour_{months}m_{stat}_{name}"
        for months in (6, 24)
        for stat in ("mean_log", "occurrence", "observed")
        for name in TARGET_NAMES
    ),
    "eligible_neighbour_fraction",
)


@dataclass(frozen=True)
class Neighbourhood:
    split: Split
    indices: np.ndarray
    ring_sizes: np.ndarray

    @classmethod
    def from_split(cls, split: Split) -> Neighbourhood:
        """Use fixed geometry only; never borrow buffer or reserved-test cells.

        Training/temporal cells use training neighbours. Geographic validation
        cells use validation neighbours' observed histories, never their futures.
        The focal cell and cells outside the fixed domain are excluded.
        """
        positions = {cell: i for i, cell in enumerate(split.cells)}
        rings = [sorted(set(h3.grid_disk(c, 1)) - {c}) for c in split.cells]
        indices = np.full((len(split.cells), max(map(len, rings))), -1, dtype=int)
        for i, (cell, ring) in enumerate(zip(split.cells, rings, strict=True)):
            group = split.groups[cell]
            if group not in {"train", "validation"}:
                continue
            neighbours = [
                positions[n]
                for n in ring
                if n in positions and split.groups[n] == group
            ]
            indices[i, : len(neighbours)] = neighbours
        return cls(split, indices, np.array([len(r) for r in rings]))

    def summaries(
        self, dataset: WindowDataset, rows: np.ndarray, device: torch.device
    ) -> torch.Tensor:
        """Average observed neighbour-months, preserving signs and missingness.

        Transform each observation before averaging. Occurrence uses individual
        abs(raw) >= 1 events, so opposite signed changes cannot cancel events.
        Missing observations never enter a mean or its denominator. Coverage
        fractions distinguish observed zero, missing history and absent neighbours.
        """
        split = dataset.corpus.split
        if (
            split.cells != self.split.cells
            or split.groups != self.split.groups
            or split.input_months != 24
            or rows.ndim != 1
            or ((rows < 0) | (rows >= len(dataset))).any()
        ):
            raise ValueError("Neighbour inputs require aligned 24-month windows.")
        cells = np.asarray(dataset.positions)[rows // len(dataset.starts)]
        if any(
            split.groups[split.cells[c]] not in {"train", "validation"} for c in cells
        ):
            raise ValueError("Neighbour inputs cannot evaluate reserved-test cells.")
        starts = np.asarray(dataset.starts)[rows % len(dataset.starts)]
        if any(
            split.months[s + split.target_months - 1] >= split.temporal["test"][0]
            for s in starts
        ):
            raise ValueError(
                "Neighbour experiment cannot evaluate reserved-test months."
            )
        months = starts[:, None] + np.arange(-24, 0)
        sums = torch.zeros(
            (len(rows), 24, len(FEATURE_NAMES)), device=device, dtype=torch.float64
        )
        counts = torch.zeros_like(sums)
        events = torch.zeros(
            (len(rows), 24, len(TARGET_NAMES)), device=device, dtype=torch.float64
        )
        neighbours = self.indices[cells]
        for slot in range(neighbours.shape[1]):
            selected = np.flatnonzero(neighbours[:, slot] >= 0)
            if not len(selected):
                continue
            source = neighbours[selected, slot, None]
            raw = torch.as_tensor(
                dataset.corpus.values[source, months[selected]], device=device
            ).double()
            mask = torch.as_tensor(
                dataset.corpus.available[source, months[selected]], device=device
            )
            target_rows = torch.as_tensor(selected, device=device)
            sums[target_rows] += torch.where(mask, transform(raw), 0)
            counts[target_rows] += mask
            events[target_rows] += (raw[:, :, TARGET_COLUMNS].abs() >= 1) & mask[
                :, :, TARGET_COLUMNS
            ]
        eligible = torch.as_tensor((neighbours >= 0).sum(1), device=device).double()
        parts = [
            sums[:, -1, STATE_COLUMNS] / counts[:, -1, STATE_COLUMNS].clamp_min(1),
            counts[:, -1, STATE_COLUMNS] / eligible[:, None].clamp_min(1),
        ]
        for lookback in (6, 24):
            observed = counts[:, -lookback:, TARGET_COLUMNS].sum(1)
            parts.extend(
                (
                    sums[:, -lookback:, TARGET_COLUMNS].sum(1) / observed.clamp_min(1),
                    events[:, -lookback:].sum(1) / observed.clamp_min(1),
                    observed / (eligible[:, None] * lookback).clamp_min(1),
                )
            )
        parts.append(
            (eligible / torch.as_tensor(self.ring_sizes[cells], device=device))[:, None]
        )
        return torch.cat(parts, dim=1)
