"""Build monthly Parquet files from one local development history extract."""

import json
import math
import resource
import time
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import h3
import osmium
import pyarrow as pa
import pyarrow.parquet as pq
from osmium.osm import Location

from atlas.features import (
    FEATURE_NAMES,
    GeometryCounts,
    build_months,
    calendar_months,
    study_cells,
)
from atlas.history import read_history


@dataclass(frozen=True)
class BuildConfig:
    history: Path
    output: Path
    bbox: tuple[float, float, float, float]
    h3_resolution: int
    start: datetime
    end: datetime
    coverage_start: datetime
    coverage_end: datetime


def _utc(value: object) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError(
            "Dates must include a UTC offset, for example 2023-01-01T00:00:00Z."
        )
    return value.astimezone(UTC)


def read_config(path: Path) -> BuildConfig:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    fields = set(BuildConfig.__dataclass_fields__) | {"purpose"}
    if set(raw) != fields or raw["purpose"] != "development":
        raise ValueError(
            "Provide the documented Milestone 1 development configuration."
        )
    bbox = raw["bbox"]
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or not all(type(v) in (int, float) and math.isfinite(v) for v in bbox)
    ):
        raise ValueError("bbox must contain four finite longitude/latitude numbers.")
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 < south < north < 90):
        raise ValueError("bbox must be ordered west, south, east, north.")
    resolution = raw["h3_resolution"]
    if type(resolution) is not int or not 0 <= resolution <= 15:
        raise ValueError("h3_resolution must be an integer between 0 and 15.")
    if not all(isinstance(raw[k], str) and raw[k] for k in ("history", "output")):
        raise ValueError("history and output must be nonempty file paths.")
    config = BuildConfig(
        history=(path.parent / raw["history"]).resolve(),
        output=(path.parent / raw["output"]).resolve(),
        bbox=(west, south, east, north),
        h3_resolution=resolution,
        start=_utc(raw["start"]),
        end=_utc(raw["end"]),
        coverage_start=_utc(raw["coverage_start"]),
        coverage_end=_utc(raw["coverage_end"]),
    )
    if config.start >= config.end or config.coverage_start >= config.coverage_end:
        raise ValueError("Each date interval must have start before end.")
    months = list(calendar_months(config.start, config.end))
    if not 1 <= len(months) <= 24:
        raise ValueError("Milestone 1 requires between 1 and 24 complete months.")
    return config


def build_dataset(config_path: Path) -> None:
    """Write one row per fixed cell/month and print the small build summary."""
    started = time.perf_counter()
    config = read_config(config_path)
    cells = study_cells(config.bbox, config.h3_resolution)
    if not 1 <= len(cells) <= 100:
        raise ValueError("Milestone 1 requires between 1 and 100 fixed study cells.")
    bounds = osmium.FileProcessor(config.history).header.box()
    if not bounds.valid() or any(
        not bounds.contains(Location(lon, lat))
        for cell in cells
        for lat, lon in h3.cell_to_boundary(cell)
    ):
        raise ValueError(
            "Study cells must fit inside the history file's declared bounds."
        )
    if config.output.exists():
        raise FileExistsError(
            f"Output already exists; choose a new directory: {config.output}"
        )
    print(f"Reading development history: {config.history}")
    history = read_history(config.history, config.start, config.end)
    print(
        f"Loaded {len(history.versions):,} entities; building {len(cells)} fixed cells."
    )
    config.output.mkdir(parents=True, exist_ok=False)
    counts = GeometryCounts()
    schema = pa.schema(
        [
            pa.field("cell", pa.string(), nullable=False),
            pa.field("month", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("available", pa.bool_(), nullable=False),
            *[
                pa.field(
                    name, pa.float64() if name.endswith(("_m", "_m2")) else pa.int64()
                )
                for name in FEATURE_NAMES
            ],
        ]
    )
    for month in build_months(
        history,
        cells,
        config.start,
        config.end,
        config.h3_resolution,
        config.coverage_start,
        config.coverage_end,
        counts,
    ):
        rows = [
            {
                "cell": cell,
                "month": month.start,
                "available": month.available,
                **{
                    name: value if month.available else None
                    for name, value in values.items()
                },
            }
            for cell, values in month.cells.items()
        ]
        destination = config.output / f"{month.start:%Y-%m}.parquet"
        with destination.open("xb") as stream:
            pq.write_table(
                pa.Table.from_pylist(rows, schema=schema), stream, compression="zstd"
            )
        print(
            f"Wrote {destination.name}: {len(rows)} rows; available={month.available}"
        )
    metadata = {
        "purpose": "development",
        "source": str(config.history),
        "start": config.start.isoformat(),
        "end": config.end.isoformat(),
        "coverage_start": config.coverage_start.isoformat(),
        "coverage_end": config.coverage_end.isoformat(),
        "bbox": config.bbox,
        "h3_resolution": config.h3_resolution,
        "cells": sorted(cells),
        "taxonomy": "atlas-m1-v1",
        "feature_columns": FEATURE_NAMES,
        "availability": "available applies to every feature in the row",
    }
    with (config.output / "dataset.json").open("x") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    print(f"Attempted geometry assignments: {counts.attempted:,}")
    print(f"Skipped geometry assignments: {counts.skipped:,}")
    print(f"Elapsed: {time.perf_counter() - started:.1f} s")
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"Peak memory: {peak_mib:.1f} MiB")
