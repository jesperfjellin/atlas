"""Compact, file-backed node and way histories for bounded geometry batches."""

from bisect import bisect_left
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

from atlas.history import Key, Version

NODE_DTYPE = np.dtype(
    [
        ("time", "<i8"),
        ("version", "<i4"),
        ("visible", "?"),
        ("lon", "<f8"),
        ("lat", "<f8"),
    ]
)
WAY_DTYPE = np.dtype(
    [
        ("time", "<i8"),
        ("version", "<i4"),
        ("visible", "?"),
        ("start", "<i8"),
        ("stop", "<i8"),
    ]
)


def save_references(
    directory: Path, kind: str, versions: dict[Key, list[Version]]
) -> None:
    """Store only geometry dependencies; contributor data and tags are not copied."""
    keys = sorted(versions)
    stem = directory / f"{kind}-{keys[-1][1]:020d}"
    index_path = stem.with_suffix(".npz")
    if index_path.exists():
        return
    offsets = [0]
    rows = []
    refs: list[int] = []
    for key in keys:
        for v in versions[key]:
            common = (int(v.timestamp.timestamp()), v.number, v.visible)
            if kind == "n":
                rows.append((*common, *(v.location or (np.nan, np.nan))))
            else:
                lower = len(refs)
                refs.extend(v.nodes)
                rows.append((*common, lower, len(refs)))
        offsets.append(len(rows))
    for suffix, array in (
        (
            ".records.npy",
            np.array(rows, dtype=NODE_DTYPE if kind == "n" else WAY_DTYPE),
        ),
        (".refs.npy", np.array(refs, dtype=np.int64)),
    ):
        if kind == "n" and suffix == ".refs.npy":
            continue
        destination = stem.with_suffix(suffix)
        partial = destination.with_suffix(".partial")
        with partial.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        partial.replace(destination)
    partial = index_path.with_suffix(".partial")
    with partial.open("wb") as stream:
        np.savez(
            stream,
            ids=np.array([key[1] for key in keys], dtype=np.int64),
            offsets=np.array(offsets, dtype=np.int64),
        )
    partial.replace(index_path)


class References:
    """Resolve IDs through small sorted indexes and memory-mapped geometry arrays."""

    def __init__(self, directory: Path, kind: str) -> None:
        self.paths = sorted(directory.glob(f"{kind}-*.npz"))
        self.ends = [int(p.stem[2:]) for p in self.paths]
        self.kind = kind

    @staticmethod
    @lru_cache(maxsize=8)
    def _arrays(
        path: Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with np.load(path, allow_pickle=False) as index:
            ids, offsets = index["ids"], index["offsets"]
        records = np.load(
            path.with_suffix(".records.npy"), mmap_mode="r", allow_pickle=False
        )
        refs = (
            np.load(path.with_suffix(".refs.npy"), mmap_mode="r", allow_pickle=False)
            if path.stem.startswith("w-")
            else np.empty(0, dtype=np.int64)
        )
        return ids, offsets, records, refs

    def load(
        self, requested: set[int], *, max_versions: int = 2_000_000
    ) -> dict[Key, list[Version]]:
        """Load a bounded dependency set, keeping missing source references absent."""
        by_shard: dict[int, list[int]] = {}
        for identifier in sorted(requested):
            shard = bisect_left(self.ends, identifier)
            if shard < len(self.paths):
                by_shard.setdefault(shard, []).append(identifier)
        result: dict[Key, list[Version]] = {}
        total = 0
        for shard, wanted in by_shard.items():
            ids, offsets, rows, refs = self._arrays(self.paths[shard])
            for identifier, position in zip(
                wanted, np.searchsorted(ids, wanted), strict=True
            ):
                if position == len(ids) or ids[position] != identifier:
                    continue
                lower, upper = int(offsets[position]), int(offsets[position + 1])
                total += upper - lower
                if total > max_versions:
                    raise MemoryError(
                        "Geometry batch exceeds two million dependency versions."
                    )
                key = (self.kind, identifier)
                versions = []
                for row in rows[lower:upper]:
                    location = None
                    nodes = ()
                    if self.kind == "n" and np.isfinite(row["lon"]):
                        location = (float(row["lon"]), float(row["lat"]))
                    elif self.kind == "w":
                        nodes = tuple(int(n) for n in refs[row["start"] : row["stop"]])
                    versions.append(
                        Version(
                            key,
                            int(row["version"]),
                            datetime.fromtimestamp(int(row["time"]), UTC),
                            bool(row["visible"]),
                            location=location,
                            nodes=nodes,
                        )
                    )
                result[key] = versions
        return result
