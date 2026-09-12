"""Accumulate independent entity histories without rebuilding unchanged snapshots."""

from bisect import bisect_right
from datetime import datetime
from itertools import groupby

import numpy as np

from atlas.features import (
    FEATURE_NAMES,
    STATE_NAMES,
    Assignment,
    GeometryCounts,
    assign,
)
from atlas.history import History, Key, Version

STATE_COLUMNS = [FEATURE_NAMES.index(f"state_{name}") for name in STATE_NAMES]
NET_COLUMNS = [FEATURE_NAMES.index(f"net_{name}") for name in STATE_NAMES]
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


def state_vector(assignment: Assignment) -> np.ndarray:
    result = np.zeros(len(STATE_NAMES))
    result[0] = 1
    for category in assignment.categories:
        result[STATE_NAMES.index(f"{category}_count")] = 1
    if "building" in assignment.categories:
        result[STATE_NAMES.index("building_area_m2")] = assignment.area_m2
    if "landuse" in assignment.categories:
        result[STATE_NAMES.index("landuse_area_m2")] = assignment.area_m2
    if any(c.startswith("road_") for c in assignment.categories):
        result[STATE_NAMES.index("road_length_m")] = assignment.length_m
    return result


class Accumulator:
    """Sparse cell arrays containing independent snapshots and change counts."""

    def __init__(
        self,
        cells: set[str],
        months: tuple[datetime, ...],
        end: datetime,
        resolution: int,
    ) -> None:
        self.cells = cells
        self.months = months
        self.end = end
        self.resolution = resolution
        self.values: dict[str, np.ndarray] = {}
        self.initial: dict[str, np.ndarray] = {}
        self.counts = GeometryCounts()

    def _values(self, cell: str) -> np.ndarray:
        if cell not in self.values:
            self.values[cell] = np.zeros((len(self.months), len(FEATURE_NAMES)))
        return self.values[cell]

    def _state(self, assignment: Assignment | None, start: int, stop: int) -> None:
        if start == stop or assignment is None or assignment.cell not in self.cells:
            return
        vector = state_vector(assignment)
        if start < 0:
            if assignment.cell not in self.initial:
                self.initial[assignment.cell] = np.zeros(len(STATE_NAMES))
            self.initial[assignment.cell] += vector
            self._values(assignment.cell)
        else:
            values = self._values(assignment.cell)
            for column in np.flatnonzero(vector):
                values[start:stop, STATE_COLUMNS[column]] += vector[column]

    def _assign(
        self,
        history: History,
        version: Version | None,
        time: datetime,
        *,
        inclusive: bool,
    ) -> Assignment | None:
        if version is None or not version.visible:
            return None
        return assign(history, version, time, self.resolution, inclusive=inclusive)

    def add(self, history: History, roots: set[Key]) -> None:
        """Count roots exactly once; referenced entities supply geometry only."""
        current: dict[Key, Assignment | None] = {}
        starts = dict.fromkeys(roots, 0)
        for key in roots:
            version = history.at(key, self.months[0], inclusive=False)
            assigned = self._assign(history, version, self.months[0], inclusive=False)
            current[key] = assigned
            if version is not None and version.visible:
                self.counts.attempted += 1
                self.counts.skipped += assigned is None
            self._state(assigned, -1, 0)

        transitions = (
            t
            for t in history.transitions(self.months[0], self.end)
            if t.after.key in roots
        )
        for time, group in groupby(transitions, key=lambda t: t.timestamp):
            month = bisect_right(self.months, time) - 1
            # Multiple versions can share a timestamp. Raw/semantic transitions
            # retain each version, but monthly state uses the final visible state.
            affected: dict[Key, Assignment | None] = {}
            for transition in group:
                before, after = transition.before, transition.after
                visible = before is not None and before.visible
                new = self._assign(history, after, time, inclusive=True)
                affected[after.key] = new
                if not visible and not after.visible:
                    continue
                self.counts.attempted += 1
                old = self._assign(history, before, time, inclusive=False)
                if (
                    not transition.complete
                    or (visible and old is None)
                    or (after.visible and new is None)
                ):
                    self.counts.skipped += 1
                    continue
                if transition.direct:
                    event = (
                        "modify"
                        if visible and after.visible
                        else ("delete" if visible else "create")
                    )
                    destination = new if after.visible else old
                    if destination is not None and destination.cell in self.cells:
                        self._values(destination.cell)[
                            month, FEATURE_INDEX[f"edit_{event}"]
                        ] += 1
                old_members = {(old.cell, c) for c in old.categories} if old else set()
                new_members = {(new.cell, c) for c in new.categories} if new else set()
                for action, memberships in (
                    ("remove", old_members - new_members),
                    ("add", new_members - old_members),
                ):
                    for cell, category in memberships:
                        if cell in self.cells:
                            self._values(cell)[
                                month, FEATURE_INDEX[f"semantic_{action}_{category}"]
                            ] += 1
            for key, assigned in affected.items():
                self._state(current[key], starts[key], month)
                starts[key] = month
                current[key] = assigned
        for key, assigned in current.items():
            self._state(assigned, starts[key], len(self.months))

    def finish(self) -> tuple[tuple[str, ...], np.ndarray]:
        cells = tuple(sorted(self.values))
        result = np.zeros((len(cells), len(self.months), len(FEATURE_NAMES)))
        for index, cell in enumerate(cells):
            values = self.values[cell]
            result[index] = values
            state = values[:, STATE_COLUMNS]
            opening = self.initial.get(cell, np.zeros(len(STATE_NAMES)))
            result[index][:, NET_COLUMNS] = np.diff(
                state, axis=0, prepend=opening[None, :]
            )
        return cells, result
