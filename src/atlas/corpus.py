"""Build the Norway corpus in resumable entity batches with complete references."""

import json
import resource
import time
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from atlas.accumulate import NET_COLUMNS, STATE_COLUMNS, Accumulator
from atlas.features import FEATURE_NAMES, STATE_NAMES, GeometryCounts, next_month
from atlas.history import History, Key, Version, iter_histories
from atlas.references import References, save_references
from atlas.splits import read_split, utc_date


def build_partitions(
    source: Path,
    output: Path,
    cells: tuple[str, ...],
    months: tuple[datetime, ...],
    resolution: int,
    batch_sizes: tuple[int, int, int] = (250_000, 10_000, 250),
) -> None:
    """Complete node, way and relation batches; publish each only after it succeeds."""
    references = output / "references"
    partitions = output / "parts"
    references.mkdir(parents=True, exist_ok=True)
    partitions.mkdir(exist_ok=True)
    start, end = months[0], next_month(months[-1])
    definition = {
        "source": str(source.resolve()),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "cells": cells,
        "h3_resolution": resolution,
        "feature_columns": FEATURE_NAMES,
    }
    definition = json.loads(json.dumps(definition))
    definition_path = output / "build.json"
    if definition_path.exists():
        previous = json.loads(definition_path.read_text())
        if any(previous.get(key) != value for key, value in definition.items()):
            raise ValueError(
                "Existing partitions use a different source or configuration."
            )
    else:
        with definition_path.open("x") as stream:
            json.dump(definition, stream)

    for kind, batch_size in zip("nwr", batch_sizes, strict=True):
        if (partitions / f"{kind}.complete").exists():
            continue
        if batch_size < 1:
            raise ValueError("Entity batch sizes must be positive.")
        node_references = References(references, "n")
        way_references = References(references, "w")
        completed = sorted(partitions.glob(f"{kind}-*.npz"))
        last_id = int(completed[-1].stem[2:]) if completed else -1

        def process(
            roots: dict[Key, list[Version]],
            kind: str = kind,
            node_references: References = node_references,
            way_references: References = way_references,
        ) -> None:
            started = time.perf_counter()
            too_large = False
            try:
                versions = roots.copy()
                if kind == "r":
                    versions.update(
                        way_references.load(
                            {
                                ref
                                for records in roots.values()
                                for v in records
                                for member_kind, ref, _ in v.members
                                if member_kind == "w"
                            }
                        )
                    )
                if kind != "n":
                    node_ids = {
                        ref
                        for records in versions.values()
                        for v in records
                        for ref in v.nodes
                    }
                    node_ids.update(
                        ref
                        for records in roots.values()
                        for v in records
                        for member_kind, ref, _ in v.members
                        if member_kind == "n"
                    )
                    versions.update(node_references.load(node_ids))
            except MemoryError:
                too_large = True
            if too_large:
                del versions
                if len(roots) == 1:
                    raise RuntimeError(
                        f"Entity exceeds the geometry budget: {next(iter(roots))}"
                    )
                keys = list(roots)
                midpoint = len(keys) // 2
                process({key: roots[key] for key in keys[:midpoint]})
                process({key: roots[key] for key in keys[midpoint:]})
                return
            accumulator = Accumulator(set(cells), months, end, resolution)
            accumulator.add(History(versions), set(roots))
            if kind in "nw":
                save_references(references, kind, roots)
            names, values = accumulator.finish()
            initial = np.array(
                [
                    accumulator.initial.get(cell, np.zeros(len(STATE_NAMES)))
                    for cell in names
                ]
            ).reshape(len(names), len(STATE_NAMES))
            destination = partitions / f"{kind}-{max(k[1] for k in roots):020d}.npz"
            partial = destination.with_suffix(".partial")
            with partial.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    cells=np.array(names, dtype="U15"),
                    values=values,
                    initial=initial,
                    attempted=accumulator.counts.attempted,
                    skipped=accumulator.counts.skipped,
                )
            partial.replace(destination)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(
                f"Completed {destination.stem}: {len(roots):,} entities, "
                f"{len(versions):,} with references; "
                f"{time.perf_counter() - started:.1f} s; peak {peak:.0f} MiB",
                flush=True,
            )

        batch: dict[Key, list[Version]] = {}
        for key, records in iter_histories(source, start, end, kind, after_id=last_id):
            batch[key] = records
            if len(batch) >= batch_size:
                process(batch)
                batch = {}
        if batch:
            process(batch)
        (partitions / f"{kind}.complete").touch(exist_ok=False)


def write_corpus(
    output: Path,
    cells: tuple[str, ...],
    months: tuple[datetime, ...],
    coverage_start: datetime,
    coverage_end: datetime,
) -> GeometryCounts:
    """Sum disjoint roots, then write every fixed cell, including empty cells."""
    if not all((output / "parts" / f"{kind}.complete").exists() for kind in "nwr"):
        raise ValueError(
            "All entity types must finish before publishing corpus months."
        )
    definition_path = output / "build.json"
    definition = json.loads(definition_path.read_text())
    coverage = {
        "coverage_start": coverage_start.isoformat(),
        "coverage_end": coverage_end.isoformat(),
    }
    if any(
        key in definition and definition[key] != value
        for key, value in coverage.items()
    ):
        raise ValueError("Existing corpus months use different source coverage.")
    definition.update(coverage)
    partial_definition = definition_path.with_suffix(".partial")
    partial_definition.write_text(json.dumps(definition))
    partial_definition.replace(definition_path)
    values = np.zeros((len(cells), len(months), len(FEATURE_NAMES)))
    initial = np.zeros((len(cells), len(STATE_NAMES)))
    index = {cell: i for i, cell in enumerate(cells)}
    counts = GeometryCounts()
    for path in sorted((output / "parts").glob("*.npz")):
        with np.load(path, allow_pickle=False) as part:
            positions = [index[str(cell)] for cell in part["cells"]]
            values[positions] += part["values"]
            initial[positions] += part["initial"]
            counts.attempted += int(part["attempted"])
            counts.skipped += int(part["skipped"])
    state = values[:, :, STATE_COLUMNS]
    values[:, :, NET_COLUMNS] = np.diff(state, axis=1, prepend=initial[:, None, :])
    integer_columns = [
        i for i, name in enumerate(FEATURE_NAMES) if not name.endswith(("_m", "_m2"))
    ]
    if (
        not np.isfinite(values).all()
        or not np.equal(
            values[:, :, integer_columns], np.rint(values[:, :, integer_columns])
        ).all()
    ):
        raise ValueError("Corpus contains non-finite values or non-integral counts.")
    for i, month in enumerate(months):
        available = coverage_start <= month and next_month(month) <= coverage_end
        columns = {
            "cell": pa.array(cells, type=pa.string()),
            "month": pa.array([month] * len(cells), type=pa.timestamp("us", tz="UTC")),
            "available": pa.array([available] * len(cells)),
        }
        for column, name in enumerate(FEATURE_NAMES):
            dtype = pa.float64() if name.endswith(("_m", "_m2")) else pa.int64()
            columns[name] = pa.array(
                values[:, i, column] if available else [None] * len(cells), type=dtype
            )
        destination = output / f"{month:%Y-%m}.parquet"
        if destination.exists():
            continue
        partial = destination.with_suffix(".partial")
        pq.write_table(pa.table(columns), partial, compression="zstd")
        partial.replace(destination)
    return counts


def build_corpus(config_path: Path) -> None:
    """Run the approved corpus configuration, resuming completed entity batches."""
    started = time.perf_counter()
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    fields = {
        "purpose",
        "history",
        "output",
        "split",
        "coverage_start",
        "coverage_end",
        "node_batch_size",
        "way_batch_size",
        "relation_batch_size",
    }
    if set(config) != fields or config["purpose"] != "corpus":
        raise ValueError("Provide the documented Norway corpus configuration.")
    split_path = (config_path.parent / config["split"]).resolve()
    split = read_split(split_path)
    source = (config_path.parent / config["history"]).resolve()
    output = (config_path.parent / config["output"]).resolve()
    coverage_start = utc_date(config["coverage_start"])
    coverage_end = utc_date(config["coverage_end"])
    if coverage_start >= coverage_end:
        raise ValueError("Source coverage must have start before end.")
    build_partitions(
        source,
        output,
        split.cells,
        split.months,
        split.resolution,
        tuple(config[f"{kind}_batch_size"] for kind in ("node", "way", "relation")),
    )
    counts = write_corpus(
        output, split.cells, split.months, coverage_start, coverage_end
    )
    metadata = {
        **json.loads((output / "build.json").read_text()),
        "taxonomy": "atlas-m1-v1",
        "split": str(split_path),
        "coverage_start": coverage_start.isoformat(),
        "coverage_end": coverage_end.isoformat(),
        "availability": "available applies to every feature in the row",
    }
    destination = output / "dataset.json"
    if not destination.exists():
        with destination.open("x") as stream:
            json.dump(metadata, stream, indent=2)
            stream.write("\n")
    disk = sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) / 1024**3
    print(
        f"Corpus: {len(split.cells):,} cells x {len(split.months)} months. "
        f"Attempted: {counts.attempted:,}; skipped: {counts.skipped:,}. "
        f"Elapsed: {time.perf_counter() - started:.1f} s; "
        f"Output including reference arrays: {disk:.2f} GiB",
        flush=True,
    )
