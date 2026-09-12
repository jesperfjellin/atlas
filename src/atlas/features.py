"""OSM category, primary-cell and calendar-month feature calculations."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise

import h3
from pyproj import Geod
from shapely.errors import GEOSException
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.geometry.polygon import orient

from atlas.history import History, Version

CATEGORIES = (
    "building",
    "road_major",
    "road_local",
    "road_path",
    "road_other",
    "poi_food",
    "poi_retail",
    "poi_service",
    "poi_other",
    "landuse",
)
MEASURES = ("building_area_m2", "road_length_m", "landuse_area_m2")
STATE_NAMES = ("entity_count", *(f"{c}_count" for c in CATEGORIES), *MEASURES)
FEATURE_NAMES = (
    "edit_create",
    "edit_modify",
    "edit_delete",
    *(f"semantic_{action}_{c}" for action in ("add", "remove") for c in CATEGORIES),
    *(f"{prefix}_{name}" for prefix in ("state", "net") for name in STATE_NAMES),
)
GEOD = Geod(ellps="WGS84")
type Values = dict[str, int | float]


@dataclass
class GeometryCounts:
    attempted: int = 0
    skipped: int = 0


@dataclass(frozen=True)
class Assignment:
    cell: str
    categories: frozenset[str]
    area_m2: float
    length_m: float


@dataclass
class Month:
    start: datetime
    available: bool
    cells: dict[str, Values]


def study_cells(bbox: tuple[float, float, float, float], resolution: int) -> set[str]:
    """Select cells by centre inside a fixed boundary, independent of OSM activity."""
    return set(h3.geo_to_cells(box(*bbox).__geo_interface__, resolution))


def categories(version: Version) -> frozenset[str]:
    tags = version.tags
    result: set[str] = set()
    if tags.get("building", "no") not in {"", "no"}:
        result.add("building")
    if tags.get("landuse", "no") not in {"", "no"}:
        result.add("landuse")
    highway = tags.get("highway", "")
    if highway and version.key[0] == "w":
        if highway in {
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "motorway_link",
            "trunk_link",
            "primary_link",
            "secondary_link",
            "tertiary_link",
        }:
            road = "major"
        elif highway in {"residential", "unclassified", "living_street", "service"}:
            road = "local"
        elif highway in {"path", "footway", "cycleway", "bridleway", "steps", "track"}:
            road = "path"
        else:
            road = "other"
        result.add(f"road_{road}")
    amenity = tags.get("amenity", "")
    if tags.get("shop", "no") not in {"", "no"}:
        result.add("poi_retail")
    elif amenity in {"restaurant", "cafe", "fast_food", "bar", "pub", "food_court"}:
        result.add("poi_food")
    elif amenity in {
        "school",
        "kindergarten",
        "university",
        "college",
        "hospital",
        "clinic",
        "doctors",
        "dentist",
        "pharmacy",
        "bank",
        "post_office",
        "library",
        "police",
        "fire_station",
        "townhall",
    }:
        result.add("poi_service")
    elif (
        amenity not in {"", "no"}
        or any(tags.get(tag, "no") not in {"", "no"} for tag in ("tourism", "leisure"))
        or highway == "bus_stop"
    ):
        result.add("poi_other")
    return frozenset(result)


def assign(
    history: History,
    version: Version,
    time: datetime,
    resolution: int,
    *,
    inclusive: bool,
) -> Assignment | None:
    if version.key[0] == "n":
        if version.location is None:
            return None
        lon, lat = version.location
        return Assignment(
            h3.latlng_to_cell(lat, lon, resolution), categories(version), 0, 0
        )
    try:
        geometry = history.geometry(version, time, inclusive=inclusive)
    except GEOSException:
        # Malformed source topology is an unavailable geometry, not a build failure.
        return None
    area, length = 0.0, 0.0
    if isinstance(geometry, Point):
        point = geometry
    elif isinstance(geometry, LineString):
        # Use distances in metres, not angular degrees, to find the midpoint.
        coordinates = list(geometry.coords)
        segments: list[tuple[float, float, float, float]] = []
        for (lon, lat), (next_lon, next_lat) in pairwise(coordinates):
            azimuth, _, distance = GEOD.inv(lon, lat, next_lon, next_lat)
            segments.append((lon, lat, azimuth, distance))
            length += distance
        remaining = length / 2
        point = Point(coordinates[-1])
        for lon, lat, azimuth, distance in segments:
            if distance >= remaining:
                x, y, _ = GEOD.fwd(lon, lat, azimuth, remaining)
                point = Point(x, y)
                break
            remaining -= distance
    elif isinstance(geometry, Polygon | MultiPolygon):
        # Ring orientation ensures that holes subtract from geodesic area.
        parts = (geometry,) if isinstance(geometry, Polygon) else geometry.geoms
        area = sum(abs(GEOD.geometry_area_perimeter(orient(p))[0]) for p in parts)
        point = geometry.representative_point()
    else:
        return None
    return Assignment(
        h3.latlng_to_cell(point.y, point.x, resolution),
        categories(version),
        area,
        length,
    )


def next_month(time: datetime) -> datetime:
    return time.replace(
        year=time.year + (time.month == 12),
        month=time.month % 12 + 1,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )


def calendar_months(start: datetime, end: datetime) -> Iterator[datetime]:
    """Yield only full calendar months inside the requested interval."""
    current = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if current < start:
        current = next_month(current)
    while next_month(current) <= end:
        yield current
        current = next_month(current)


def snapshot(
    history: History,
    cells: set[str],
    time: datetime,
    resolution: int,
    counts: GeometryCounts,
) -> dict[str, Values]:
    result: dict[str, Values] = {cell: dict.fromkeys(STATE_NAMES, 0) for cell in cells}
    for key in history.versions:
        version = history.at(key, time, inclusive=False)
        if version is None or not version.visible:
            continue
        counts.attempted += 1
        assigned = assign(history, version, time, resolution, inclusive=False)
        if assigned is None:
            counts.skipped += 1
            continue
        if assigned.cell not in cells:
            continue
        values = result[assigned.cell]
        values["entity_count"] += 1
        for category in assigned.categories:
            values[f"{category}_count"] += 1
        if "building" in assigned.categories:
            values["building_area_m2"] += assigned.area_m2
        if "landuse" in assigned.categories:
            values["landuse_area_m2"] += assigned.area_m2
        if any(c.startswith("road_") for c in assigned.categories):
            values["road_length_m"] += assigned.length_m
    return result


def build_months(
    history: History,
    cells: set[str],
    start: datetime,
    end: datetime,
    resolution: int,
    coverage_start: datetime,
    coverage_end: datetime,
    counts: GeometryCounts,
) -> Iterator[Month]:
    previous: dict[str, Values] | None = None
    for month in calendar_months(start, end):
        stop = next_month(month)
        values: dict[str, Values] = {
            cell: dict.fromkeys(FEATURE_NAMES, 0) for cell in sorted(cells)
        }
        if month < coverage_start or stop > coverage_end:
            previous = None
            yield Month(month, False, values)
            continue
        if previous is None:
            previous = snapshot(history, cells, month, resolution, counts)
        current = snapshot(history, cells, stop, resolution, counts)
        for cell in cells:
            for name in STATE_NAMES:
                values[cell][f"state_{name}"] = current[cell][name]
                values[cell][f"net_{name}"] = current[cell][name] - previous[cell][name]
        for transition in history.transitions(month, stop):
            before, after = transition.before, transition.after
            was_visible = before is not None and before.visible
            if not was_visible and not after.visible:
                continue
            counts.attempted += 1  # A transition is one paired geometry assignment.
            old = (
                assign(
                    history, before, transition.timestamp, resolution, inclusive=False
                )
                if before is not None and was_visible
                else None
            )
            new = (
                assign(history, after, transition.timestamp, resolution, inclusive=True)
                if after.visible
                else None
            )
            if (
                not transition.complete
                or (was_visible and old is None)
                or (after.visible and new is None)
            ):
                counts.skipped += 1
                continue
            if transition.direct:
                event = (
                    "modify"
                    if was_visible and after.visible
                    else ("delete" if was_visible else "create")
                )
                destination = new if after.visible else old
                if destination is not None and destination.cell in cells:
                    values[destination.cell][f"edit_{event}"] += 1
            old_members = {(old.cell, c) for c in old.categories} if old else set()
            new_members = {(new.cell, c) for c in new.categories} if new else set()
            for action, memberships in (
                ("remove", old_members - new_members),
                ("add", new_members - old_members),
            ):
                for cell, category in memberships:
                    if cell in cells:
                        values[cell][f"semantic_{action}_{category}"] += 1
        yield Month(month, True, values)
        previous = current
