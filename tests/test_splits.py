"""Protect temporal cutoffs and geographic isolation in the actual experiment."""

from datetime import UTC, datetime
from pathlib import Path

import h3

from atlas.splits import read_split


def test_fixed_splits_keep_neighbours_and_complete_target_windows_separate() -> None:
    split = read_split(Path(__file__).parents[1] / "configs/split.yaml")
    groups = {
        name: set(split.eligible_cells(name, geographic=name != "train"))
        for name in ("train", "validation", "test")
    }
    assert all(groups.values())
    assert not (
        groups["train"] & groups["validation"]
        or groups["train"] & groups["test"]
        or groups["validation"] & groups["test"]
    )
    for cell in split.cells:
        if split.groups[cell] == "buffer":
            continue
        for neighbour in h3.grid_disk(cell, 1):
            if neighbour in split.groups:
                assert split.groups[neighbour] in {"buffer", split.groups[cell]}
    for geographic in (False, True):
        assert (
            not set(split.eligible_cells("test", geographic=geographic))
            & split.development_cells
        )
    for partition in ("train", "validation", "test"):
        starts = split.target_starts(partition)
        assert starts
        lower, upper = split.temporal[partition]
        for start in starts:
            inputs = split.months[start - split.input_months : start]
            targets = split.months[start : start + split.target_months]
            assert len(inputs) == 24 and len(targets) == 6
            assert inputs[-1] < targets[0]
            assert lower <= targets[0] <= targets[-1] < upper
    first = split.target_starts("validation")[0]
    assert split.months[first - 24] == datetime(2021, 1, 1, tzinfo=UTC)
    assert split.months[first] == datetime(2023, 1, 1, tzinfo=UTC)
    assert all(split.months[i].month <= 7 for i in split.target_starts("test"))
