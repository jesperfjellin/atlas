# Atlas

Atlas is a small experiment that learns from changes in OpenStreetMap in Norway.
The current implementation provides the GPU runtime scaffold for Milestone 0.
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
The single automated smoke test checks `atlas --help`. GPU acceptance is a manual Compose check.

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

The scaffold includes no OSM data or model results.
Git excludes `data/` and `runs/`. The Docker image excludes data, credentials, checkpoints, and caches.

OSM data attribution: © OpenStreetMap contributors.
OpenStreetMap data is available under the [Open Database License (ODbL)](https://www.openstreetmap.org/copyright).
Data use and redistribution must comply with that license and retain the required attribution.
