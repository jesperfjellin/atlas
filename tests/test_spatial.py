"""Protect neighbour membership, temporal cutoffs and observed denominators."""

from datetime import UTC, datetime

import h3
import numpy as np
import pytest
import torch

from atlas.features import FEATURE_NAMES, calendar_months
from atlas.linear import historical_split
from atlas.samples import CellMonths, Preprocessing, WindowDataset
from atlas.spatial import NEIGHBOUR_NAMES, Neighbourhood
from atlas.splits import Split


def spatial_corpus(resolution: int) -> tuple[CellMonths, str, str, str]:
    focal = h3.latlng_to_cell(59.9, 10.7, resolution)
    geographic = h3.latlng_to_cell(63.4, 10.4, resolution)
    isolated = h3.latlng_to_cell(70, 25, resolution)
    groups = {focal: "train", geographic: "validation", isolated: "train"}
    ring = sorted(set(h3.grid_disk(focal, 1)) - {focal})
    # Leave one geometric neighbour outside the corpus.
    groups.update(
        zip(ring[:5], ("train", "train", "validation", "test", "buffer"), strict=True)
    )
    ring = sorted(set(h3.grid_disk(geographic, 1)) - {geographic})
    groups.update(
        zip(
            ring,
            ("validation", "validation", "validation", "train", "test", "buffer"),
            strict=True,
        )
    )
    cells = tuple(sorted(groups))

    def date(year: int) -> datetime:
        return datetime(year, 1, 1, tzinfo=UTC)

    months = tuple(calendar_months(date(2015), date(2026)))
    split = Split(
        cells,
        months,
        groups,
        frozenset(),
        {
            "train": (date(2017), date(2023)),
            "validation": (date(2023), date(2025)),
            "test": (date(2025), date(2026)),
        },
        24,
        6,
        resolution,
    )
    values = np.zeros((len(cells), len(months), len(FEATURE_NAMES)), dtype=np.float32)
    return (
        CellMonths(split, values, np.ones_like(values, dtype=bool)),
        focal,
        geographic,
        isolated,
    )


@pytest.mark.parametrize("resolution", [5, 7])
def test_neighbours_preserve_geographic_and_historical_training_boundaries(
    resolution: int,
) -> None:
    corpus, focal, geographic, _ = spatial_corpus(resolution)
    neighbours = Neighbourhood.from_split(corpus.split)
    device = torch.tensor(0).device
    for cell, count in ((focal, 2), (geographic, 3)):
        actual = neighbours.indices[corpus.split.cells.index(cell)]
        actual = actual[actual >= 0]
        assert len(actual) == count
        assert all(
            corpus.split.groups[corpus.split.cells[n]] == corpus.split.groups[cell]
            for n in actual
        )
        assert corpus.split.cells.index(cell) not in actual
    for year in (2020, 2021, 2022):
        fold = CellMonths(
            historical_split(corpus.split, year),
            corpus.values.copy(),
            corpus.available.copy(),
        )
        prep = Preprocessing.fit(fold)
        training = WindowDataset(fold, prep, "train")
        rows = np.arange(len(training))
        before = neighbours.summaries(training, rows, device)
        # Even later *training target* months cannot influence input scaling.
        fold.values[:, training.starts[-1] :] = 1e9
        fold.available[:, training.starts[-1] :] = False
        for i, cell in enumerate(fold.split.cells):
            if fold.split.groups[cell] != "train":
                fold.values[i] = -1e9
                fold.available[i] = False
        after = neighbours.summaries(training, rows, device)
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    prep = Preprocessing.fit(corpus)
    validation = WindowDataset(corpus, prep, "validation", geographic=True)
    position = validation.positions.index(corpus.split.cells.index(geographic))
    rows = np.array([position * len(validation.starts)])
    before = neighbours.summaries(validation, rows, device)
    corpus.values[:, validation.starts[0] :] = 1e9
    for i, cell in enumerate(corpus.split.cells):
        if corpus.split.groups[cell] != "validation":
            corpus.values[i] = 1e9
    torch.testing.assert_close(neighbours.summaries(validation, rows, device), before)
    for heldout in (
        WindowDataset(corpus, prep, "test"),
        WindowDataset(corpus, prep, "test", geographic=True),
    ):
        with pytest.raises(ValueError):
            neighbours.summaries(heldout, np.array([0]), device)


def test_neighbour_means_preserve_events_missingness_and_empty_rings() -> None:
    corpus, focal, _, isolated = spatial_corpus(6)
    neighbours = Neighbourhood.from_split(corpus.split)
    position = corpus.split.cells.index(focal)
    a, b = neighbours.indices[position, :2]
    change = FEATURE_NAMES.index("net_building_count")
    state = FEATURE_NAMES.index("state_building_count")
    corpus.values[a, :24, change] = 3
    corpus.values[b, :24, change] = -3
    corpus.available[a, 18:21, change] = False
    corpus.values[a, 18:21, change] = np.nan
    corpus.values[a, 23, state] = 8
    corpus.values[b, 23, state] = 0
    prep = Preprocessing(np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)))
    dataset = WindowDataset(corpus, prep, "train")
    rows = np.array(
        [
            dataset.positions.index(p) * len(dataset.starts)
            for p in (position, corpus.split.cells.index(isolated))
        ]
    )
    actual = neighbours.summaries(dataset, rows, torch.tensor(0).device)

    def value(name: str) -> float:
        return float(actual[0, NEIGHBOUR_NAMES.index(name)])

    assert value("neighbour_6m_mean_log_net_building_count") == pytest.approx(
        -np.log(4) / 3
    )
    assert value("neighbour_6m_occurrence_net_building_count") == 1
    assert value("neighbour_6m_observed_net_building_count") == 0.75
    assert value("neighbour_mean_log_state_building_count") == pytest.approx(
        np.log(9) / 2
    )
    assert value("neighbour_observed_state_building_count") == 1
    assert value("eligible_neighbour_fraction") == pytest.approx(2 / 6)
    assert value("neighbour_6m_mean_log_edit_create") == 0
    assert value("neighbour_6m_observed_edit_create") == 1
    assert not actual[1].any()  # No neighbours, distinct from observed zero above.
    corpus.available[a, 18:24, change] = False
    corpus.available[b, 18:24, change] = False
    corpus.values[position, :24] = 1e12  # Self cannot enter neighbouring history.
    actual = neighbours.summaries(dataset, rows, torch.tensor(0).device)
    assert value("neighbour_6m_mean_log_net_building_count") == 0
    assert value("neighbour_6m_occurrence_net_building_count") == 0
    assert value("neighbour_6m_observed_net_building_count") == 0
    assert torch.isfinite(actual).all()
