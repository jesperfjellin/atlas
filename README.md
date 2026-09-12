# Atlas

Atlas is a small experiment that learns from changes in OpenStreetMap in Norway.
Atlas builds monthly OSM features in bounded batches and provides the AMD GPU runtime for the ML experiment.
The [specification](SPEC.md) defines the experiment. [PROGRESS.md](PROGRESS.md) tracks its capabilities and remaining work.

## Run with Docker Compose

Use WSL2 with an AMD Windows driver and Docker with the Compose plugin.
The Docker daemon must be able to pass `/dev/dxg` into containers.
Run these commands from the repository root:

```bash
docker compose build
docker compose run --rm atlas
docker compose run --rm atlas uv run --locked atlas --help
```

The default command is `atlas doctor`.
It prints runtime versions and the AMD GPU name, then checks a small GPU tensor operation and its gradient.
It fails if ROCm, GPU access, or either calculation fails.
All ML runs use the GPU. There is no CPU mode or fallback.

The image includes the [ROCDXG bridge library](https://github.com/ROCm/librocdxg) and the system libraries needed by PyTorch.
On first use, `uv` installs the locked Python dependencies, including PyTorch for ROCm, inside the container.
Compose passes the GPU device and mounts WSL's `/usr/lib/wsl/lib/libdxcore.so` into the container.
GPU profiling is disabled because the ROCm profiler requires Linux KFD interfaces that WSL does not provide.
The runtime was checked on an RX 7900 XTX with Adrenalin 26.8.1 and Docker Desktop.

Compose mounts the repository at `/workspace`.
Named Docker volumes keep the Python environment at `/opt/venv` and the download cache at `/tmp/uv-cache` across container runs and image builds.
The service uses user and group IDs of 1000 by default.
If your IDs differ, set `ATLAS_UID` and `ATLAS_GID` to the values from `id -u` and `id -g`.
After a system dependency change, run `docker compose build` again.
Python dependencies are synchronized from `uv.lock` when commands run.

## Development

The project uses Python 3.14 and locks its dependencies in `uv.lock`.
Install dependencies and run Python tools through Compose. Torch and its dependency cache stay inside Docker.
The image includes `uv`. The development tools use the application's environment and lock file.
Tests check OSM change semantics, historical geometry, monthly coverage, and the installed CLI. GPU acceptance is a manual Compose check.

Run the checks before each handoff:

```bash
docker compose run --rm atlas uv sync --locked
docker compose run --rm atlas uv run ruff format --check .
docker compose run --rm atlas uv run ruff check .
docker compose run --rm atlas uv run ty check
docker compose run --rm atlas uv run pytest
docker compose run --rm atlas uv run pre-commit run --all-files
```

## OSM data

### Build the Norway corpus

Milestone 2 uses [configs/norway.toml](configs/norway.toml) and the fixed [split](configs/split.yaml).
The supplied full-history file must be at `data/raw/norway-internal.osh.pbf`.

```bash
docker compose run --rm atlas uv run --locked atlas build-dataset \
  --config configs/norway.toml
```

The same command resumes an interrupted build. Completed entity batches remain usable.
Node and way reference histories use ordinary NumPy files. Each entity contributes features once; shared references supply geometry only.
This preserves entities that cross processing boundaries without clipping their geometry or duplicating counts.
Monthly states reuse geometry between edits, including edits to referenced children. Net changes still come from independently reconstructed boundary states.
The two geometry counters count opening snapshots and paired transitions; cached monthly snapshots do not add assignment attempts.

The output contains monthly Parquet files, interpretation metadata, completed batch arrays, and compact reference arrays.
Do not change the source file or build configuration while a build is in progress.
The corpus includes every fixed cell in every complete month, including empty cells and geographic buffers.

The checked-in boundary comes from [Natural Earth 1:10m Admin 0 Countries](https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/).
It covers mainland Norway, Svalbard, and Jan Mayen. Bouvet Island is outside the supplied history extract and is excluded.
This approximate land boundary defines the experiment domain only. It supplies no predictive features.
H3 resolution 6 gives 14,446 study cells. Coastal cells follow this fixed boundary rather than detailed cadastral coastlines.

The split uses training targets in 2017–2022, validation in 2023–2024, and reserved temporal testing in 2025.
H3 parent groups also provide geographic validation and testing. Their cells are excluded from training at all dates.
Cells adjacent to another geographic group are buffers and cannot enter samples. Kristiansand development cells cannot enter reserved testing.
Use temporal and geographic evaluation separately; neither distribution is resampled.

[samples.py](src/atlas/samples.py) provides `CellMonths.read`, `Preprocessing.fit`, and `WindowDataset` for PyTorch's `DataLoader`.
Each sample has 24 input months and six following target months, all within one target partition.
Inputs contain 51 features, two calendar encodings, and 51 availability indicators. Targets contain the 37 change features and their masks.
Input scaling fits signed-log values from unique training input cell-months only. Losses and primary metrics remain subject to Gate 2.
Dataset loading performs structural checks without printing reserved-test target summaries.

### Build the development slice

The first slice uses Kristiansand in 2023. Its fixed boundary and dates are in [configs/kristiansand.toml](configs/kristiansand.toml).
This area and period, including earlier reconstruction history, are development data. They must not enter the reserved test partition.

Obtain a full-history `norway-internal.osh.pbf` from [Geofabrik](https://download.geofabrik.de/europe/norway.html) and place it in `data/raw/`.
The download requires an OpenStreetMap login. A current-state `.osm.pbf` file cannot replace the history file.

After rebuilding the image, create the local extract if it does not already exist:

```bash
docker compose build
docker compose run --rm atlas osmium extract \
  --with-history --bbox 7.75,57.95,8.30,58.35 --set-bounds \
  data/raw/norway-internal.osh.pbf --output data/raw/kristiansand.osh.pbf
docker compose run --rm atlas uv run --locked atlas build-dataset \
  --config configs/kristiansand.toml
```

The extraction boundary includes a buffer around the study cells. Osmium retains full ways and their referenced nodes.
Relations can remain incomplete; the builder omits geometries it cannot reconstruct.
Extraction does not clip geometry or build Norway-wide features. See the [Osmium history extraction rules](https://docs.osmcode.org/osmium/latest/osmium-extract.html).

The builder writes one Parquet file per complete UTC month and a small `dataset.json` file in the configured output directory.
Each Parquet row contains a fixed H3 cell, a month, an availability flag, and the feature columns.
The same cells appear every month, including empty cells. Paths in the configuration are relative to that file.
All study cells must fit inside the history file's declared extraction bounds.
Existing output directories are protected from overwrite. To repeat a build, choose a new output directory in the configuration.
Milestone 1 limits builds to 100 cells, 24 months, and a 256 MiB input extract.

`coverage_start` and `coverage_end` declare the source interval. Do not infer coverage from the first or last observed edit.
The supplied history file lacks a replication timestamp header; the example declares only the coverage needed for this development slice.
Requested partial calendar months are excluded. A whole month outside source coverage has null features and `available=false`.
Skipped individual geometries leave the reconstructable subset numeric, including zero. The availability flag applies to every feature in the row.

### Feature meaning

The three change families follow [SPEC.md, Section 7](SPEC.md#7-exact-meaning-of-change):

- `edit_create`, `edit_modify`, and `edit_delete` count entity version transitions, including untagged nodes.
- `semantic_add_*` and `semantic_remove_*` count category membership changes and movement between primary cells.
- `net_*` is the difference between independently reconstructed states at the two month boundaries. `state_*` contains the closing state.

Boundary snapshots use the state immediately before the boundary. An edit exactly at midnight on the first day belongs to the new month.
Referenced nodes and ways are resolved at the relevant historical time. Child edits can change parent geometry and semantic cell assignment without adding parent raw edits.
When geometry becomes reconstructable or unavailable, net state change need not equal semantic additions minus removals.
OSM timestamps have one-second precision. Reference lookup uses the last version in that second for the after-state and excludes that second for the before-state.

The small taxonomy is defined in [features.py](src/atlas/features.py): buildings, four road groups, four POI groups, and land use.
Road categories apply to ways. One POI category is selected in this order: retail, food, services, other.
An entity may also be a building or land-use feature. `building=no` and `landuse=no` are excluded.
Mapped state includes entity and category counts, building and land-use area in square metres, and road length in metres.

Nodes use their coordinates. Lines use a midpoint along geodesic segment lengths. Polygons use a point on their surface.
Area and length use WGS84 geodesic measurements. Whole entities are assigned to one cell; geometry is not split across cells.
The relation assembler supports complete way-member multipolygons with disjoint outer rings and ordinary holes.
It skips nested same-role rings, nested relation members, unsupported relation types, missing references, and invalid topology.
Gaps in entity version history also cause the affected transition to be skipped.

The console reports runtime, peak memory, attempted geometry assignments, and skipped geometry assignments.
Each visible snapshot is one assignment attempt. Each direct or child-induced transition is one paired assignment attempt.
Counters include candidates outside the study cells that occur in the buffered history extract.

### Attribution and storage

The repository includes a synthetic fixture, but no real OSM data or model results.
Git excludes `data/` and `runs/`. The Docker image excludes data, credentials, checkpoints, and caches.

OSM data attribution: © OpenStreetMap contributors.
OpenStreetMap data is available under the [Open Database License (ODbL)](https://www.openstreetmap.org/copyright).
Data use and redistribution must comply with that license and retain the required attribution.
