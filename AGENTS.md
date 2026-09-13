# Agent Instructions

These instructions apply to the entire repository.

## Read first

- Read the project specification before editing (`SPEC.md` in the repository; some copies may be named `spec.md`).
- Read `PROGRESS.md` for the product roadmap, current capabilities, remaining work, model evidence, and blockers.
- Keep H3 resolution configurable. For an approved cell-size change, follow [the resolution-change guide](docs/cell-resolution.md); do not hardcode the current grid's cell count into model code.
- Work only on the currently approved milestone.
- Follow **LOCKED**, **PROVISIONAL**, and **DEFERRED** decisions exactly as defined there.
- Keep the existing Python package and CLI name `atlas`.
- If these instructions conflict with the specification, follow the specification and report the conflict.

## Maintain the product roadmap

- Keep `PROGRESS.md` as the single product roadmap.
- If capabilities, evidence, remaining work, or blockers change, update the relevant sections in place.
- Mark a capability complete only after its required behavior exists and its relevant checks pass.
- Distinguish implemented capabilities from demonstrated model results.
- Link to `SPEC.md` for contracts and gates instead of duplicating the specification.
- Keep chronological activity, document-edit history, and routine command results out of `PROGRESS.md`.
- Put routine check results in the completion report. Retain only evidence or blockers that affect the roadmap.

## Prime directive: keep the proof of concept lean

The ML experiment is the project. OSM preparation exists only to feed it.

- Implement the smallest direct solution that runs the current experiment.
- Do not build an audit system, provenance framework, data catalogue, experiment registry, workflow engine, job queue, plugin system, generic service layer, or report generator.
- Do not add abstractions for hypothetical future backends, sources, models, regions, or deployment modes.
- Do not create placeholder modules or commands for deferred milestones.
- Do not write planning documents or ADRs unless the owner asks for one.
- Prefer ordinary files, functions, and small modules over frameworks.
- If new support code does not unblock a current requirement, do not add it.

## Working safely

- Inspect the relevant code, tests, configuration, and Git status before editing.
- Preserve unrelated user changes and keep the patch narrowly scoped.
- Commit completed work directly to `main` in meaningful chunks after the required checks pass.
- Group related changes into one commit. Do not create a commit for every minor edit.
- Push completed commits to `origin/main` without separate confirmation, unless the owner explicitly asks to keep changes local.
- Do not delete data or run destructive Git commands unless explicitly asked.
- Do not start a full Norway data build unless the relevant gate has been approved.
- Never hide a failing check or silently weaken a test to make it pass.

## Python toolchain

- Use the Python version pinned by `.python-version` and `pyproject.toml`.
- Use `uv` for Python, environments, dependencies, locking, and command execution.
- Install dependencies and run Python tools inside the Compose service, using `uv`.
- Keep Torch execution and its dependency cache inside Docker. Do not install or cache Torch on the host.
- Use `uv add <package>` for runtime dependencies.
- Use `uv add --dev <package>` for development dependencies.
- Use `uv remove <package>` to remove dependencies.
- Commit `uv.lock`; never edit it manually.
- Do not introduce pip, Poetry, Pipenv, Conda, or a hand-maintained requirements workflow.

Common commands:

```bash
docker compose run --rm atlas uv sync --locked
docker compose run --rm atlas uv run atlas --help
docker compose run --rm atlas uv run pytest
docker compose run --rm atlas uv run ty check
docker compose run --rm atlas uv run ruff check .
docker compose run --rm atlas uv run ruff format --check .
docker compose run --rm atlas uv run pre-commit run --all-files
```

## Code

- Use Ruff for formatting and linting.
- Use `ty` for static type checking.
- Type public interfaces and non-obvious data structures.
- Prefer `pathlib.Path`, timezone-aware UTC datetimes, explicit errors, and simple logging.
- Keep I/O at the edges so OSM transformations and ML logic can be tested directly.
- Validate external data and configuration at their boundaries.
- Preserve the distinction between observed zero and unavailable data.
- Avoid broad `Any`, blanket suppressions, premature base classes, factories, and registries.
- Add dependencies only when they directly simplify a current requirement.

## Test policy

Reserve automated tests for durable project logic where a regression could silently corrupt data, invalidate an ML result, or break a reused public contract.

Prioritize tests for:

- OSM temporal and spatial transformations;
- change-counting semantics;
- train/validation/test leakage rules;
- dataset shapes and target alignment;
- model forward/loss/checkpoint behavior;
- regressions for bugs that have actually occurred.

Do not normally add tests for:

- diagnostic or `doctor` commands beyond one basic smoke test;
- mocked GPU, ROCm, CUDA, driver, filesystem, or host capability detection;
- exact CLI output or error-message wording;
- one-off environment checks;
- trivial branches, getters, configuration plumbing, or pass-through wrappers;
- behavior owned by Python, PyTorch, or another dependency;
- implementation details that are likely to change during the proof of concept.

Verification is not synonymous with an automated test.
Running a command manually and reporting its result is sufficient for environment-dependent behavior.

Default test budget:

- infrastructure, diagnostics, and glue changes: zero new tests unless fixing a regression;
- small ordinary changes: at most one focused test;
- core OSM or ML logic: as many tests as needed to protect the important invariants.

Before exceeding the default budget, explain what realistic regression each additional test prevents.

## Running tests

- Use pytest.
- Run the narrowest relevant test while developing:

```bash
docker compose run --rm atlas uv run pytest tests/path/to/test_file.py::test_name
```

- Do not chase coverage percentage or test trivial implementation details.
- Unit tests must not require network access, credentials, large downloads, or a GPU.
- Keep slow, real-data, and accelerator checks opt-in.
- Use small synthetic fixtures with exact expected values.
- Before handoff, run the complete unit-test suite.

## ML and data safeguards

These rules protect the validity of the proof of concept and are not optional auditing:

- Never select cells or evaluation samples using future OSM activity.
- Never let future months enter model inputs.
- Use complete UTC calendar months with `[start, end)` boundaries.
- Assign events at a month boundary to the new month.
- Keep each sample's full target window inside one temporal partition.
- Permit earlier historical input months as specified in `SPEC.md`, subject to the sample cutoff and geographic holdouts.
- Omit snapshots and transitions with unreconstructable geometry from cell features.
- If individual snapshots or transitions are skipped, keep computed features numeric for the reconstructable entity subset.
- Reserve `null` and a false availability mask for an uncomputable cell-month feature, such as missing source coverage.
- Record only attempted and skipped geometry assignments as build counters.
- Fit normalization and learned preprocessing on training data only.
- Do not inspect reserved-test target distributions or model results while choosing features, losses, thresholds, or hyperparameters.
- Keep validation and test distributions unresampled.
- Save the resolved training configuration, metrics, and checkpoints in a simple run directory.
- Do not add a database or lineage system for this metadata.

## Docker

- Supported workflows must run through Docker Compose.
- Start with one reusable image and service.
- Configure AMD ROCm in the main Compose service. Do not add CPU/GPU profiles or overrides.
- ML commands require the AMD GPU and must fail if it is unavailable. Do not implement CPU fallback.
- Validate GPU execution inside Compose with `atlas doctor`. Do not request GPU device access for the agent process.
- Do not bake data, credentials, checkpoints, or caches into the image.

## Before handoff

Run from the repository root:

```bash
docker compose run --rm atlas uv sync --locked
docker compose run --rm atlas uv run ruff format --check .
docker compose run --rm atlas uv run ruff check .
docker compose run --rm atlas uv run ty check
docker compose run --rm atlas uv run pytest
docker compose run --rm atlas uv run pre-commit run --all-files
```

Run `docker compose run --rm atlas` for GPU runtime changes. If a check cannot run, give the exact command and reason.

Keep the completion report short:

- what changed;
- checks run and results;
- any deviation or blocker;
- the next decision, if one is actually needed.
