"""Fixed study cells and leakage-safe temporal and geographic sample selection."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import h3
import yaml
from shapely.geometry import shape

from atlas.features import calendar_months


def utc_date(value: str) -> datetime:
    time = datetime.fromisoformat(value)
    if time.tzinfo is None:
        time = time.replace(tzinfo=UTC)
    time = time.astimezone(UTC)
    if (time.day, time.hour, time.minute, time.second, time.microsecond) != (
        1,
        0,
        0,
        0,
        0,
    ):
        raise ValueError("Split dates must be UTC calendar-month boundaries.")
    return time


@dataclass(frozen=True)
class Split:
    cells: tuple[str, ...]
    months: tuple[datetime, ...]
    groups: dict[str, str]
    development_cells: frozenset[str]
    temporal: dict[str, tuple[datetime, datetime]]
    input_months: int
    target_months: int
    resolution: int

    def eligible_cells(
        self, partition: str, *, geographic: bool = False
    ) -> tuple[str, ...]:
        if partition not in self.temporal or (partition == "train" and geographic):
            raise ValueError(
                "Use train, validation or test. "
                "Geographic holdouts are evaluation only."
            )
        group = partition if geographic else "train"
        return tuple(
            cell
            for cell in self.cells
            if self.groups[cell] == group
            and not (partition == "test" and cell in self.development_cells)
        )

    def target_starts(self, partition: str) -> tuple[int, ...]:
        start, end = self.temporal[partition]
        return tuple(
            index
            for index in range(
                self.input_months, len(self.months) - self.target_months + 1
            )
            if self.months[index] >= start
            and self.months[index + self.target_months - 1] < end
        )


def read_split(path: Path) -> Split:
    """Read the fixed boundary and remove neighbours across geographic splits."""
    raw = yaml.safe_load(path.read_text())
    boundary = json.loads((path.parent / raw["boundary"]).read_text())["geometry"]
    geometry = shape(boundary)
    if (
        not geometry.is_valid
        or geometry.is_empty
        or geometry.geom_type not in {"Polygon", "MultiPolygon"}
    ):
        raise ValueError("The study boundary must be a valid polygon or multipolygon.")
    resolution = raw["h3_resolution"]
    cells = tuple(sorted(h3.geo_to_cells(boundary, resolution)))
    geo = raw["geographic"]
    validation = set(geo["validation_parents"])
    test = set(geo["test_parents"])
    parent_resolution = geo["parent_resolution"]
    if validation & test or not 0 <= parent_resolution < resolution <= 15:
        raise ValueError(
            "Geographic holdout parents must be disjoint and coarser than study cells."
        )
    if any(
        not h3.is_valid_cell(p) or h3.get_resolution(p) != parent_resolution
        for p in validation | test
    ):
        raise ValueError("Invalid geographic holdout parent.")
    if geo["buffer_rings"] < 1:
        raise ValueError("At least one ring must separate geographic partitions.")

    def group(cell: str) -> str:
        parent = h3.cell_to_parent(cell, parent_resolution)
        return (
            "test"
            if parent in test
            else ("validation" if parent in validation else "train")
        )

    # Classify neighbours beyond the land boundary too, so narrow coastlines do
    # not bypass the geographic gap. Buffer rows remain in the corpus.
    groups = {
        cell: (
            "buffer"
            if any(
                group(n) != group(cell) for n in h3.grid_disk(cell, geo["buffer_rings"])
            )
            else group(cell)
        )
        for cell in cells
    }
    development = frozenset(geo["development_cells"])
    if any(
        h3.cell_to_parent(c, parent_resolution) in validation | test
        for c in development
    ):
        raise ValueError(
            "Previously inspected development cells cannot become geographic holdouts."
        )
    months = tuple(calendar_months(utc_date(raw["start"]), utc_date(raw["end"])))
    temporal = {
        name: (utc_date(a), utc_date(b)) for name, (a, b) in raw["temporal"].items()
    }
    if set(temporal) != {"train", "validation", "test"} or not months:
        raise ValueError(
            "Provide train, validation and test intervals within the corpus."
        )
    intervals = [temporal[name] for name in ("train", "validation", "test")]
    if any(a >= b for a, b in intervals) or any(
        intervals[i][1] > intervals[i + 1][0] for i in (0, 1)
    ):
        raise ValueError("Temporal partitions must be ordered and non-overlapping.")
    if intervals[0][0] < months[0] or intervals[-1][1] > utc_date(raw["end"]):
        raise ValueError("Temporal partitions must fit within corpus coverage.")
    if any(
        type(raw[k]) is not int or raw[k] < 1 for k in ("input_months", "target_months")
    ):
        raise ValueError(
            "Input and target windows must have positive whole month counts."
        )
    return Split(
        cells,
        months,
        groups,
        development,
        temporal,
        raw["input_months"],
        raw["target_months"],
        resolution,
    )
