"""Read a small OSM history extract and resolve references at historical times."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import groupby, pairwise
from pathlib import Path

import osmium
import osmium.osm
from osmium.osm import Node, Relation, Way
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize_full, unary_union

type Key = tuple[str, int]


@dataclass(frozen=True, slots=True)
class Version:
    key: Key
    number: int
    timestamp: datetime
    visible: bool
    tags: dict[str, str] = field(default_factory=dict)
    location: tuple[float, float] | None = None  # longitude, latitude
    nodes: tuple[int, ...] = ()
    members: tuple[tuple[str, int, str], ...] = ()

    def references(self) -> Iterator[Key]:
        yield from (("n", ref) for ref in self.nodes)
        yield from ((kind, ref) for kind, ref, _ in self.members)


@dataclass(frozen=True, slots=True)
class Transition:
    before: Version | None
    after: Version
    timestamp: datetime
    direct: bool
    complete: bool = True


class History:
    """In-memory versions for the bounded Milestone 1 extract."""

    def __init__(self, versions: dict[Key, list[Version]]) -> None:
        self.versions = versions
        self.parents: dict[Key, set[Key]] = defaultdict(set)
        for key, records in versions.items():
            for previous, current in pairwise(records):
                if (
                    previous.number >= current.number
                    or previous.timestamp > current.timestamp
                ):
                    raise ValueError(f"History is not ordered by version/time: {key}")
            for version in records:
                for ref in version.references():
                    self.parents[ref].add(key)

    def at(self, key: Key, time: datetime, *, inclusive: bool) -> Version | None:
        records = self.versions.get(key, [])
        search = bisect_right if inclusive else bisect_left
        index = search(records, time, key=lambda v: v.timestamp) - 1
        return records[index] if index >= 0 else None

    def transitions(self, start: datetime, end: datetime) -> Iterator[Transition]:
        """Include child geometry changes without inventing parent OSM edits."""
        events: dict[datetime, list[Transition]] = defaultdict(list)
        for records in self.versions.values():
            previous = None
            for current in records:
                if start <= current.timestamp < end:
                    complete = current.number == (
                        previous.number + 1 if previous else 1
                    )
                    events[current.timestamp].append(
                        Transition(previous, current, current.timestamp, True, complete)
                    )
                previous = current
        for time, direct in sorted(events.items()):
            yield from direct
            affected = {event.after.key for event in direct}
            pending = list(affected)
            while pending:
                child = pending.pop()
                for parent in self.parents.get(child, ()):
                    if parent in affected:
                        continue
                    before = self.at(parent, time, inclusive=False)
                    after = self.at(parent, time, inclusive=True)
                    if after is None or not any(
                        version is not None and child in version.references()
                        for version in (before, after)
                    ):
                        continue
                    affected.add(parent)
                    pending.append(parent)
                    yield Transition(before, after, time, False)

    def geometry(
        self, version: Version, time: datetime, *, inclusive: bool
    ) -> BaseGeometry | None:
        """Reconstruct nodes, ways and simple way-member multipolygons."""
        if not version.visible:
            return None
        if version.key[0] == "n":
            return Point(version.location) if version.location is not None else None
        if version.key[0] == "w":
            line = self._line(version, time, inclusive=inclusive)
            if line is None:
                return None
            tags = version.tags
            area = tags.get("area") != "no" and (
                tags.get("area") == "yes"
                or any(tags.get(tag, "no") != "no" for tag in ("building", "landuse"))
                or (
                    "highway" not in tags
                    and any(
                        tag in tags for tag in ("amenity", "shop", "tourism", "leisure")
                    )
                )
            )
            if area and version.nodes[0] == version.nodes[-1]:
                if len(version.nodes) < 4:
                    return None
                polygon = Polygon(line.coords)
                return polygon if polygon.is_valid and polygon.area > 0 else None
            return line
        if version.tags.get("type") not in {"multipolygon", "boundary"}:
            return None
        rings: dict[str, list[LineString]] = {"outer": [], "inner": []}
        for kind, ref, role in version.members:
            if role not in {"", "outer", "inner"}:
                if kind == "n" and role in {"label", "admin_centre"}:
                    continue
                return None
            member = self.at((kind, ref), time, inclusive=inclusive)
            if kind != "w" or member is None or not member.visible:
                return None
            line = self._line(member, time, inclusive=inclusive)
            if line is None:
                return None
            rings[role or "outer"].append(line)
        if not rings["outer"]:
            return None
        polygons: dict[str, BaseGeometry] = {}
        for role, lines in rings.items():
            assembled, cuts, dangles, invalid = polygonize_full(lines)
            if not (cuts.is_empty and dangles.is_empty and invalid.is_empty):
                return None
            # Nested rings of the same role need more than this simple assembler.
            if any(polygon.interiors for polygon in assembled.geoms):
                return None
            polygons[role] = unary_union(assembled)
        outer, inner = polygons["outer"], polygons["inner"]
        if not inner.is_empty and not outer.covers(inner):
            return None
        result = outer.difference(inner)
        if (
            isinstance(result, Polygon | MultiPolygon)
            and result.is_valid
            and not result.is_empty
        ):
            return result
        return None

    def _line(
        self, version: Version, time: datetime, *, inclusive: bool
    ) -> LineString | None:
        points: list[tuple[float, float]] = []
        for ref in version.nodes:
            node = self.at(("n", ref), time, inclusive=inclusive)
            if node is None or not node.visible or node.location is None:
                return None
            points.append(node.location)
        if len(points) < 2:
            return None
        line = LineString(points)
        return line if line.is_valid and line.length > 0 else None


def read_history(path: Path, start: datetime, end: datetime) -> History:
    """Keep the last pre-start version and in-window versions, without user data."""
    if path.stat().st_size > 256 * 1024**2:
        raise ValueError("Milestone 1 requires an extract smaller than 256 MiB.")
    return History(dict(iter_histories(path, start, end)))


def iter_histories(
    path: Path,
    start: datetime,
    end: datetime,
    kind: str | None = None,
    *,
    after_id: int = -1,
) -> Iterator[tuple[Key, list[Version]]]:
    """Stream one entity at a time from a type/ID/version-sorted history file."""
    entities = {
        None: osmium.osm.ALL,
        "n": osmium.osm.NODE,
        "w": osmium.osm.WAY,
        "r": osmium.osm.RELATION,
    }[kind]
    processor = osmium.FileProcessor(path, entities=entities)
    if not processor.header.has_multiple_object_versions:
        raise ValueError("Expected a full-history file, not an OSM snapshot.")
    previous_key = (-1, -1)
    for key, objects in groupby(processor, key=lambda obj: (obj.type_str(), obj.id)):
        ordered_key = ("nwr".index(key[0]), key[1])
        if ordered_key <= previous_key:
            raise ValueError("History must be sorted by entity type and ID.")
        previous_key = ordered_key
        if key[1] <= after_id:
            continue
        records: list[Version] = []
        for obj in objects:
            if not isinstance(obj, Node | Way | Relation):
                raise ValueError(
                    "Expected only nodes, ways and relations in OSM history."
                )
            time = obj.timestamp.astimezone(UTC)
            if time >= end:
                continue
            if obj.version < 1 or time.year < 2000:
                raise ValueError("OSM history requires version numbers and timestamps.")
            location = None
            nodes = ()
            members = ()
            if isinstance(obj, Node) and obj.visible and obj.location.valid():
                location = (obj.location.lon, obj.location.lat)
            elif isinstance(obj, Way):
                nodes = tuple(node.ref for node in obj.nodes)
            elif isinstance(obj, Relation):
                members = tuple((m.type, m.ref, m.role) for m in obj.members)
            version = Version(
                key,
                obj.version,
                time,
                obj.visible,
                dict(obj.tags),
                location,
                nodes,
                members,
            )
            if records and (
                records[-1].number >= version.number or records[-1].timestamp > time
            ):
                raise ValueError(f"History is not ordered by version/time: {key}")
            if time < start:
                records[:] = [version]
            else:
                records.append(version)
        if records:
            yield key, records
