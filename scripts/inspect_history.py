"""Replay the selected development histories with Atlas's production accumulator.

After diagnose_activity.py, run through Compose from the repository root:
    uv run python scripts/inspect_history.py --case case-01
    uv run python scripts/inspect_history.py

The default processes the complete manifest, reusing completed cases. Extracts
are buffered candidate sets, not guaranteed censuses of primary-cell entities.
Reconciliation against the original corpus exposes that limitation explicitly.
"""

import argparse
import csv
import json
import resource
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import h3
import numpy as np
import osmium
import pyarrow as pa
import pyarrow.parquet as pq
from osmium.osm import Node, Relation, Way
from shapely.errors import GEOSException

from atlas.accumulate import Accumulator
from atlas.baselines import write_json
from atlas.features import (
    FEATURE_NAMES,
    assign,
    calendar_months,
    categories,
    next_month,
)
from atlas.history import History, Key, Version, iter_histories
from atlas.references import References
from atlas.splits import read_split, utc_date

CORPUS = Path("data/derived/norway-2015-2025")
SOURCE = Path("data/raw/norway-internal.osh.pbf")
DEFAULT_OUTPUT = Path("runs/history-diagnostics")


class ReconciliationRow(TypedDict):
    cell: str
    month: str
    feature: str
    original: float
    replay: float
    difference: float
    matches: bool


EVENT_SCHEMA = pa.schema(
    [
        *[
            (n, pa.string())
            for n in (
                "case",
                "entity",
                "timestamp",
                "scope",
                "event",
                "role",
                "status",
                "old_cell",
                "new_cell",
                "old_categories",
                "new_categories",
                "flags",
                "old_tags",
                "new_tags",
                "old_geometry",
                "new_geometry",
            )
        ],
        *[
            (n, pa.int64())
            for n in (
                "before_version",
                "after_version",
                "changeset",
                "old_vertices",
                "new_vertices",
            )
        ],
        ("direct", pa.bool_()),
        *[
            (n, pa.float64())
            for n in ("old_area_m2", "new_area_m2", "old_length_m", "new_length_m")
        ],
    ]
)


def command_output(destination: Path, command: list[str]) -> None:
    if destination.exists():
        return
    temporary = destination.with_suffix(".partial")
    subprocess.run([*command, "-f", "osh.pbf", "-O", "-o", str(temporary)], check=True)
    temporary.replace(destination)


def trim(records: list[Version], start: datetime, end: datetime) -> list[Version]:
    result: list[Version] = []
    for v in records:
        if v.timestamp >= end:
            continue
        if v.timestamp < start:
            result[:] = [v]
        else:
            result.append(v)
    return result


def geometry_text(
    history: History, version: Version | None, time: datetime, inclusive: bool
) -> str | None:
    if version is None or not version.visible:
        return None
    try:
        geometry = history.geometry(version, time, inclusive=inclusive)
    except GEOSException:
        return None
    return geometry.wkt if geometry is not None else None


def inspect_case(case: dict[str, str], output: Path, source: Path) -> None:
    split = read_split(Path("configs/split.yaml"))
    cell, focal = case["cell"], utc_date(case["month"])
    if (
        split.groups.get(cell) != "train"
        or not split.months[0] <= focal < split.temporal["test"][0]
    ):
        raise ValueError(
            "Inspection cases must be development months in training geography."
        )
    directory = output / case["case"]
    directory.mkdir(exist_ok=True)
    definition = {
        "case": case,
        "source": str(SOURCE.resolve()),
        "test_start": split.temporal["test"][0].isoformat(),
        "buffer_rings": 1,
    }
    config = directory / "case.json"
    if config.exists() and json.loads(config.read_text()) != definition:
        raise ValueError("Case selection changed; use a new output directory.")
    write_json(config, definition)
    if (directory / "summary.json").exists():
        print(f"Reusing {case['case']}.", flush=True)
        return
    started = time.monotonic()
    ring = set(h3.grid_disk(cell, 1))
    # Context outside training geography can supply references, never diagnostic
    # target rows. Versions after the case's context are also excluded below.
    context = {c for c in ring if split.groups.get(c) == "train"}
    bounds = [point for c in ring for point in h3.cell_to_boundary(c)]
    bbox = (
        min(p[1] for p in bounds) - 0.001,
        min(p[0] for p in bounds) - 0.001,
        max(p[1] for p in bounds) + 0.001,
        max(p[0] for p in bounds) + 0.001,
    )
    extract = directory / "history.osh.pbf"
    print(
        f"Extracting {case['case']}: {cell}, {focal:%Y-%m}, {case['stratum']}.",
        flush=True,
    )
    command_output(
        extract,
        [
            "osmium",
            "extract",
            "-H",
            "-s",
            "complete_ways",
            "--clean",
            "uid",
            "--clean",
            "user",
            "-b",
            ",".join(map(str, bbox)),
            str(source),
        ],
    )
    if extract.stat().st_size > 256 * 1024**2:
        raise ValueError(
            "Case extract exceeds 256 MiB; narrow the inspection explicitly."
        )
    previous = datetime(
        focal.year - (focal.month == 1), (focal.month - 2) % 12 + 1, 1, tzinfo=UTC
    )
    start = max(previous, split.months[0])
    end = min(next_month(next_month(focal)), split.temporal["test"][0])
    months = tuple(calendar_months(start, end))
    base = History(dict(iter_histories(extract, start, end)))
    changesets: dict[tuple[Key, int], int] = {}
    for obj in osmium.FileProcessor(extract):
        if not isinstance(obj, Node | Way | Relation):
            raise ValueError("Expected node, way and relation history records.")
        if start <= obj.timestamp < end and obj.changeset:
            changesets[((obj.type_str(), obj.id), obj.version)] = obj.changeset
    references = {k: References(CORPUS / "references", k) for k in "nw"}
    accumulator = Accumulator(context, months, end, split.resolution)
    roles: Counter[str] = Counter()
    flags_count: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    rows: list[dict] = []
    total_rows = 0
    destination = directory / "history_diagnostics.parquet"
    temporary = destination.with_suffix(".partial")

    def supplement(roots: dict[Key, list[Version]]) -> History:
        versions = roots.copy()
        ways = {
            ref
            for records in roots.values()
            for v in records
            if v.tags.get("type") in {"multipolygon", "boundary"}
            for kind, ref, _ in v.members
            if kind == "w"
        }
        # Reference arrays contain geometry only. Retain actual tags where the
        # spatial extract has that way; reference-only ways never become roots.
        for key, records in references["w"].load(ways).items():
            versions[key] = base.versions.get(key, trim(records, start, end))
        nodes = {
            ref for records in versions.values() for v in records for ref in v.nodes
        }
        nodes.update(
            ref
            for records in roots.values()
            for v in records
            for kind, ref, _ in v.members
            if kind == "n"
        )
        for key, records in references["n"].load(nodes).items():
            versions.setdefault(key, trim(records, start, end))
        return History({key: records for key, records in versions.items() if records})

    with pq.ParquetWriter(temporary, EVENT_SCHEMA, compression="zstd") as event_writer:

        def process(roots: dict[Key, list[Version]]) -> None:
            nonlocal total_rows
            try:
                history = supplement(roots)
            except MemoryError:
                if len(roots) == 1:
                    raise
                keys = list(roots)
                middle = len(keys) // 2
                process({key: roots[key] for key in keys[:middle]})
                process({key: roots[key] for key in keys[middle:]})
                return
            accumulator.add(history, set(roots))
            for transition in history.transitions(focal, next_month(focal)):
                before, after = transition.before, transition.after
                if after.key not in roots:
                    continue
                visible = before is not None and before.visible
                if not visible and not after.visible:
                    continue
                t = transition.timestamp
                old = (
                    assign(history, before, t, split.resolution, inclusive=False)
                    if visible
                    else None
                )
                new = (
                    assign(history, after, t, split.resolution, inclusive=True)
                    if after.visible
                    else None
                )
                assigned_cells = {a.cell for a in (old, new) if a is not None}
                if assigned_cells and not assigned_cells & context:
                    continue
                scope = (
                    "focal"
                    if cell in assigned_cells
                    else ("neighbor" if assigned_cells else "unlocated_extract_root")
                )
                status = (
                    "version_gap"
                    if not transition.complete
                    else "geometry_unavailable"
                    if (visible and old is None) or (after.visible and new is None)
                    else "counted"
                )
                event = (
                    "modify"
                    if visible and after.visible
                    else ("delete" if visible else "create")
                )
                representative = after if after.visible else before
                assert representative is not None
                cats = categories(representative)
                role = (
                    "classified"
                    if cats
                    else "address_node"
                    if representative.key[0] == "n"
                    and any(k.startswith("addr:") for k in representative.tags)
                    else "untagged_node"
                    if representative.key[0] == "n" and not representative.tags
                    else "other_tagged"
                    if representative.tags
                    else "other_untagged"
                )
                flags = []
                if before is not None and before.visible and after.visible:
                    geometry_changed = (
                        before.location != after.location
                        or before.nodes != after.nodes
                        or before.members != after.members
                    )
                    if before.tags != after.tags and not geometry_changed:
                        flags.append("tag_only_version")
                    if geometry_changed:
                        flags.append("geometry_reference_or_coordinate_edit")
                    if (
                        before.tags.get("landuse") == "forest"
                        and after.tags.get("natural") == "wood"
                        and "landuse" not in after.tags
                    ):
                        flags.append("landuse_forest_to_natural_wood")
                    if (
                        before.tags.get("natural") == "wood"
                        and after.tags.get("landuse") == "forest"
                        and "natural" not in after.tags
                    ):
                        flags.append("natural_wood_to_landuse_forest")
                if not transition.direct:
                    flags.append("child_induced")
                if old and new and old.cell != new.cell:
                    flags.append("primary_cell_move")
                for version in (before, after):
                    if version is None or version.tags.get("type") != "multipolygon":
                        continue
                    for kind, ref, role_name in version.members:
                        member = base.at((kind, ref), t, inclusive=version is after)
                        if (
                            kind == "w"
                            and role_name in {"", "outer"}
                            and member
                            and categories(member) - categories(version)
                        ):
                            flags.append("outer_way_category_without_relation_category")
                            break
                status_key = f"{scope}/{status}"
                statuses[status_key] += 1
                flags = sorted(set(flags))
                if scope == "focal":
                    flags_count.update(flags)
                destination_assignment = new if after.visible else old
                if (
                    status == "counted"
                    and transition.direct
                    and destination_assignment
                    and destination_assignment.cell == cell
                ):
                    roles[f"{event}/{after.key[0]}/{role}"] += 1
                rows.append(
                    {
                        "case": case["case"],
                        "entity": f"{after.key[0]}{after.key[1]}",
                        "timestamp": t.isoformat(),
                        "scope": scope,
                        "event": event,
                        "role": role,
                        "status": status,
                        "direct": transition.direct,
                        "before_version": before.number if before else None,
                        "after_version": after.number,
                        "changeset": changesets.get((after.key, after.number))
                        if transition.direct
                        else None,
                        "old_cell": old.cell if old else None,
                        "new_cell": new.cell if new else None,
                        "old_categories": json.dumps(sorted(old.categories))
                        if old
                        else None,
                        "new_categories": json.dumps(sorted(new.categories))
                        if new
                        else None,
                        "flags": json.dumps(flags),
                        "old_tags": json.dumps(before.tags, sort_keys=True)
                        if before
                        else None,
                        "new_tags": json.dumps(after.tags, sort_keys=True),
                        "old_vertices": len(before.nodes) if before else None,
                        "new_vertices": len(after.nodes),
                        "old_geometry": geometry_text(history, before, t, False),
                        "new_geometry": geometry_text(history, after, t, True),
                        "old_area_m2": old.area_m2 if old else None,
                        "new_area_m2": new.area_m2 if new else None,
                        "old_length_m": old.length_m if old else None,
                        "new_length_m": new.length_m if new else None,
                    }
                )
                total_rows += 1
                if len(rows) >= 1000:
                    event_writer.write_table(
                        pa.Table.from_pylist(rows, schema=EVENT_SCHEMA)
                    )
                    rows.clear()

        for kind, size in (("n", 10000), ("w", 500), ("r", 10)):
            keys = [key for key in base.versions if key[0] == kind]
            for offset in range(0, len(keys), size):
                process(
                    {key: base.versions[key] for key in keys[offset : offset + size]}
                )
        if rows:
            event_writer.write_table(pa.Table.from_pylist(rows, schema=EVENT_SCHEMA))
    temporary.replace(destination)
    names, computed = accumulator.finish()
    index = {c: i for i, c in enumerate(names)}
    reconciliation: list[ReconciliationRow] = []
    for j, month in enumerate(months):
        table = pq.read_table(
            CORPUS / f"{month:%Y-%m}.parquet", filters=[("cell", "in", sorted(context))]
        )
        for original in table.to_pylist():
            c = original["cell"]
            for k, feature in enumerate(FEATURE_NAMES):
                value = computed[index[c], j, k] if c in index else 0.0
                expected = original[feature]
                if expected is None:
                    raise ValueError("Inspection requires observed development rows.")
                continuous = feature.endswith(("_m", "_m2"))
                matches = bool(
                    np.isclose(value, expected, rtol=1e-8, atol=0.01)
                    if continuous
                    else value == expected
                )
                reconciliation.append(
                    {
                        "cell": c,
                        "month": month.date().isoformat(),
                        "feature": feature,
                        "original": expected,
                        "replay": value,
                        "difference": value - expected,
                        "matches": matches,
                    }
                )
    with (directory / "reconciliation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reconciliation[0]))
        writer.writeheader()
        writer.writerows(reconciliation)
    focal_rows = [
        r for r in reconciliation if r["cell"] == cell and r["month"] == case["month"]
    ]
    matches = sum(r["matches"] for r in focal_rows)
    raw_replay = sum(
        r["replay"] for r in focal_rows if r["feature"].startswith("edit_")
    )
    if raw_replay != sum(roles.values()):
        raise ValueError(
            "Diagnostic raw composition does not reconcile to the production replay."
        )
    write_json(
        directory / "summary.json",
        {
            "case": case,
            "focal_features_matching": matches,
            "feature_count": len(FEATURE_NAMES),
            "raw_composition": dict(roles),
            "focal_flags": dict(flags_count),
            "transition_statuses": dict(statuses),
            "diagnostic_rows": total_rows,
            "in_window_versions_with_changeset": len(changesets),
            "extract_bytes": extract.stat().st_size,
            "seconds": time.monotonic() - started,
            "peak_process_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "interpretation": "Flags are candidates, not attribution. "
            "Changeset comments were not fetched. Neighbor rows are context. "
            "Unlocated failures cannot be assigned to the focal cell. "
            "An extract may omit roots with distant vertices and a primary point "
            "inside the cell. Reference completion restores dependencies only. "
            "A mismatch is unresolved extraction/replay coverage until examined; "
            "it does not establish a corpus bug.",
        },
    )
    print(
        f"Completed {case['case']}: {matches}/{len(FEATURE_NAMES)} features match; "
        f"{int(raw_replay):,} raw edits; {time.monotonic() - started:.1f} s.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path("runs").resolve()):
        raise ValueError("Write diagnostic outputs beneath runs/.")
    with (args.output / "sample_manifest.csv").open() as stream:
        cases = list(csv.DictReader(stream))
    selected = [c for c in cases if not args.case or c["case"] in args.case]
    if args.case and set(args.case) - {c["case"] for c in selected}:
        raise ValueError("Unknown case ID.")
    split = read_split(Path("configs/split.yaml"))
    source = args.output / "history-before-test.osh.pbf"
    command_output(
        source,
        [
            "osmium",
            "time-filter",
            str(SOURCE),
            "2000-01-01T00:00:00Z",
            split.temporal["test"][0].strftime("%Y-%m-%dT%H:%M:%SZ"),
        ],
    )
    for case in selected:
        inspect_case(case, args.output, source)


if __name__ == "__main__":
    main()
