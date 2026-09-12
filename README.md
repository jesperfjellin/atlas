# Atlas

Atlas is a small experiment that learns from changes in OpenStreetMap in Norway.
The current implementation provides the CPU scaffold for Milestone 0.
The [specification](SPEC.md) defines the experiment. [PROGRESS.md](PROGRESS.md) tracks its capabilities and remaining work.

## Run with Docker Compose

Install Docker with the Compose plugin.
Run these commands from the repository root:

```bash
docker compose build
docker compose run --rm atlas
docker compose run --rm atlas uv run --locked atlas --help
```

The default command is `atlas doctor`.
It prints runtime versions and GPU availability, then checks a small CPU tensor operation and its gradient.
The command returns a nonzero status if the runtime check fails.
The image uses the CPU build of PyTorch. AMD compatibility checks are in progress at Gate 0.
GPU availability describes what this PyTorch installation can use.

Compose mounts the repository at `/workspace` and keeps the container environment at `/opt/venv`.
The service uses user and group IDs of 1000 by default.
If your IDs differ, set `ATLAS_UID` and `ATLAS_GID` to the values from `id -u` and `id -g`.
After a dependency change, run `docker compose build` again.

## AMD compatibility check

The optional check requires a ROCm build of PyTorch and access to an AMD GPU:

```bash
uv run atlas doctor --device rocm
```

The check prints the GPU name and checks the matrix result and gradient on that GPU.
It returns a nonzero status if ROCm or the GPU is unavailable. It never falls back to CPU.
The default CPU environment rejects this request.
An AMD Compose configuration awaits a successful device check, as required by [Gate 0](SPEC.md#milestone-0--minimal-scaffold).

## Local development

Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run these commands from the repository root:

```bash
uv sync --locked
uv run atlas doctor
uv run atlas --help
uv run pre-commit install
```

The project uses Python 3.14 and locks its dependencies in `uv.lock`.
The development tools use the same environment and lock file as the application.

Run the checks before each handoff:

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
uv run pre-commit run --all-files
```

Run the same checks in the Compose service:

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
