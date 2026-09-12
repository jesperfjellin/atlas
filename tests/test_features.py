"""Protect counting semantics and historical geometry using one small history."""

from datetime import UTC, datetime
from pathlib import Path

import h3
import numpy as np
import osmium
import osmium.io
import pyarrow.parquet as pq
import pytest
from osmium.osm import Box, Location
from shapely.geometry import Point, Polygon

from atlas.accumulate import Accumulator
from atlas.dataset import build_dataset
from atlas.features import (
    FEATURE_NAMES,
    GeometryCounts,
    assign,
    build_months,
    calendar_months,
    study_cells,
)
from atlas.history import History, read_history
from atlas.references import References, save_references

START = datetime(2023, 1, 1, tzinfo=UTC)
END = datetime(2023, 4, 1, tzinfo=UTC)
BBOX = (7.90, 58.05, 8.15, 58.25)
A = h3.latlng_to_cell(58.15, 8.0, 6)
B = h3.latlng_to_cell(58.18, 8.10, 6)


@pytest.fixture
def history(tmp_path: Path) -> History:
    path = tmp_path / "history.osh.pbf"
    header = osmium.io.Header()
    header.has_multiple_object_versions = True
    header.add_box(Box(Location(7.75, 57.95), Location(8.30, 58.35)))
    with osmium.SimpleWriter(path, header=header) as writer:
        for obj in osmium.FileProcessor(Path(__file__).parent / "fixtures/history.osh"):
            writer.add(obj)
    return read_history(path, START, END)


def test_discarded_timestamp_reversal_keeps_the_valid_opening_state(
    tmp_path: Path,
) -> None:
    # Norway way 8332583 has an 86-second reversal in March 2008, before our window.
    source = tmp_path / "reversed.osh"
    source.write_text(
        '<osm version="0.6">'
        '<way id="1" version="3" timestamp="2008-03-17T04:44:35Z" visible="true"/>'
        '<way id="1" version="4" timestamp="2008-03-17T04:43:09Z" visible="true"/>'
        '<way id="1" version="7" timestamp="2012-11-29T18:57:13Z" visible="true">'
        '<nd ref="10"/><nd ref="11"/></way>'
        '<way id="1" version="8" timestamp="2017-12-07T21:34:03Z" visible="false"/>'
        "</osm>"
    )
    path = tmp_path / "reversed.osh.pbf"
    header = osmium.io.Header()
    header.has_multiple_object_versions = True
    with osmium.SimpleWriter(path, header=header) as writer:
        for obj in osmium.FileProcessor(source):
            writer.add(obj)
    start = datetime(2015, 1, 1, tzinfo=UTC)
    retained = read_history(path, start, END)
    records = retained.versions[("w", 1)]
    assert [version.number for version in records] == [7, 8]
    assert retained.at(("w", 1), start, inclusive=False) == records[0]
    assert records[0].nodes == (10, 11)
    (transition,) = retained.transitions(start, END)
    assert transition.before == records[0] and transition.after == records[1]
    assert transition.complete and not transition.after.visible
    # Reject the same reversal when either affected version enters the timeline.
    for cutoff in (
        datetime(2008, 1, 1, tzinfo=UTC),
        datetime(2008, 3, 17, 4, 44, tzinfo=UTC),
    ):
        with pytest.raises(ValueError):
            read_history(path, cutoff, END)


def test_monthly_change_families_and_movement(history: History) -> None:
    cells = study_cells(BBOX, 6)
    counts = GeometryCounts()
    january, february, march = list(
        build_months(history, cells, START, END, 6, START, END, counts)
    )
    assert A != B and {A, B} <= cells
    assert set(january.cells) == set(february.cells) == set(march.cells) == cells
    assert all(value == 0 for value in january.cells[B].values())
    jan = january.cells[A]
    assert [jan[f"edit_{event}"] for event in ("create", "modify", "delete")] == [
        1,
        5,
        0,
    ]
    assert jan["semantic_add_poi_food"] == jan["semantic_remove_poi_food"] == 1
    assert jan["semantic_add_poi_service"] == jan["state_poi_service_count"] == 1
    assert jan["semantic_add_building"] == jan["semantic_remove_building"] == 1
    assert jan["net_building_count"] == 0
    assert jan["net_building_area_m2"] > 0
    assert jan["semantic_remove_road_local"] == jan["semantic_add_road_major"] == 1
    assert jan["net_road_length_m"] == pytest.approx(0)
    assert jan["net_entity_count"] == 1

    old, new = february.cells[A], february.cells[B]
    assert old["edit_modify"] == 1  # the road's own category change
    assert new["edit_modify"] == 5  # moved POI plus four moved building nodes
    assert old["semantic_remove_poi_service"] == new["semantic_add_poi_service"] == 1
    assert old["semantic_remove_building"] == new["semantic_add_building"] == 1
    assert old["net_building_count"] == -1 and new["net_building_count"] == 1
    assert old["semantic_remove_road_major"] == old["semantic_add_poi_food"] == 1
    assert march.cells[A]["edit_delete"] == march.cells[B]["edit_delete"] == 1
    assert march.cells[B]["edit_create"] == 1
    assert march.cells[B]["semantic_remove_poi_service"] == 1
    assert march.cells[A]["semantic_remove_poi_food"] == 1
    assert ("n", 3) not in history.versions  # an edit exactly at the interval end


def test_historical_geometry_and_multipolygon_hole(history: History) -> None:
    building = history.versions[("w", 10)][0]
    before = history.geometry(building, START, inclusive=False)
    january = history.geometry(
        building, datetime(2023, 2, 1, tzinfo=UTC), inclusive=False
    )
    assert before == Polygon(
        [(8.001, 58.151), (8.002, 58.151), (8.002, 58.152), (8.001, 58.152)]
    )
    assert january == Polygon(
        [(8.001, 58.151), (8.003, 58.151), (8.002, 58.152), (8.001, 58.152)]
    )
    relation = history.versions[("r", 40)][0]
    polygon = history.geometry(relation, START, inclusive=False)
    assert isinstance(polygon, Polygon)
    assert polygon.area == pytest.approx(0.000015)
    assert not polygon.covers(Point(8.0055, 58.1555))
    # A self-crossing source ring must not terminate a build.
    malformed = history.versions[("r", 50)][0]
    assert assign(history, malformed, START, 6, inclusive=False) is None


def test_missing_geometry_keeps_subset_numeric(history: History) -> None:
    incomplete = History({key: history.versions[key] for key in (("n", 30), ("w", 30))})
    counts = GeometryCounts()
    (month,) = build_months(
        incomplete, {A}, START, datetime(2023, 2, 1, tzinfo=UTC), 6, START, END, counts
    )
    assert month.available
    assert month.cells[A]["state_entity_count"] == 1
    assert month.cells[A]["edit_modify"] == 0
    assert month.cells[A]["state_road_path_count"] == 0
    assert all(isinstance(value, int | float) for value in month.cells[A].values())
    assert counts == GeometryCounts(attempted=5, skipped=3)
    # A missing version must not turn a larger history jump into one known edit.
    records = history.versions[("n", 2)]
    gapped = History({("n", 2): [records[0], records[-1]]})
    counts = GeometryCounts()
    (month,) = build_months(
        gapped, {A}, START, datetime(2023, 2, 1, tzinfo=UTC), 6, START, END, counts
    )
    assert month.cells[A]["edit_modify"] == 0
    assert counts == GeometryCounts(attempted=3, skipped=1)


def test_complete_calendar_months_and_source_coverage(history: History) -> None:
    assert list(
        calendar_months(
            datetime(2023, 1, 15, tzinfo=UTC), datetime(2023, 3, 15, tzinfo=UTC)
        )
    ) == [datetime(2023, 2, 1, tzinfo=UTC)]
    counts = GeometryCounts()
    months = list(
        build_months(
            history,
            {A},
            START,
            END,
            6,
            datetime(2023, 1, 15, tzinfo=UTC),
            datetime(2023, 3, 15, tzinfo=UTC),
            counts,
        )
    )
    assert [month.available for month in months] == [False, True, False]
    assert months[1].cells[A]["semantic_remove_poi_service"] == 1


def test_parquet_preserves_unavailable_values_and_empty_cells(
    history: History, tmp_path: Path
) -> None:
    assert history.versions
    config = tmp_path / "slice.toml"
    config.write_text("""purpose = "development"
history = "history.osh.pbf"
output = "output"
bbox = [7.90, 58.05, 8.15, 58.25]
h3_resolution = 6
start = 2023-01-01T00:00:00Z
end = 2023-04-01T00:00:00Z
coverage_start = 2023-02-01T00:00:00Z
coverage_end = 2023-04-01T00:00:00Z
""")
    build_dataset(config)
    january = pq.read_table(tmp_path / "output/2023-01.parquet")
    march = pq.read_table(tmp_path / "output/2023-03.parquet")
    cells = study_cells(BBOX, 6)
    assert january.num_rows == march.num_rows == len(cells)
    assert set(january.column("cell").to_pylist()) == cells
    assert set(march.column("cell").to_pylist()) == cells
    assert january.column("available").to_pylist() == [False] * len(cells)
    assert all(january.column(name).null_count == len(cells) for name in FEATURE_NAMES)
    assert all(march.column(name).null_count == 0 for name in FEATURE_NAMES)
    empty_cell = h3.latlng_to_cell(58.23, 7.91, 6)
    empty = next(row for row in march.to_pylist() if row["cell"] == empty_cell)
    assert empty["available"] and all(empty[name] == 0 for name in FEATURE_NAMES)
    assert march.schema.field("month").type.tz == "UTC"
    assert march.column("month").to_pylist() == [
        datetime(2023, 3, 1, tzinfo=UTC)
    ] * len(cells)


def test_bounded_reference_batches_equal_independent_monthly_snapshots(
    history: History, tmp_path: Path
) -> None:
    cells = study_cells(BBOX, 6)
    months = tuple(calendar_months(START, END))
    node_keys = sorted(k for k in history.versions if k[0] == "n")
    midpoint = len(node_keys) // 2
    for keys in (node_keys[:midpoint], node_keys[midpoint:]):
        save_references(tmp_path, "n", {k: history.versions[k] for k in keys})
    ways = {k: v for k, v in history.versions.items() if k[0] == "w"}
    save_references(tmp_path, "w", ways)
    nodes_on_disk, ways_on_disk = References(tmp_path, "n"), References(tmp_path, "w")
    actual = {cell: np.zeros((3, len(FEATURE_NAMES))) for cell in cells}
    # Deliberately split roots one by one, with shared references in other files.
    for key, records in history.versions.items():
        versions = {key: records}
        if key[0] == "r":
            versions.update(
                ways_on_disk.load(
                    {ref for v in records for kind, ref, _ in v.members if kind == "w"}
                )
            )
        if key[0] != "n":
            versions.update(
                nodes_on_disk.load(
                    {
                        ref
                        for records in versions.values()
                        for v in records
                        for ref in v.nodes
                    }
                )
            )
        accumulator = Accumulator(cells, months, END, 6)
        accumulator.add(History(versions), {key})
        names, values = accumulator.finish()
        for index, cell in enumerate(names):
            actual[cell] += values[index]
    expected = list(
        build_months(history, cells, START, END, 6, START, END, GeometryCounts())
    )
    for cell in cells:
        np.testing.assert_allclose(
            actual[cell],
            [[month.cells[cell][name] for name in FEATURE_NAMES] for month in expected],
            atol=1e-7,
            rtol=1e-10,
        )
