# Atlas progress

This is the single product roadmap for the Atlas OSM learning experiment.
The [specification](SPEC.md), version 0.3, defines the experiment, technical contracts, and milestone gates.
This document tracks capabilities, remaining work, model evidence, and blockers.

## Product and current position

Atlas aims to predict future changes in OpenStreetMap and learn useful representations of places through time.
The experiment uses historical OSM data from Norway and runs on one consumer desktop.
The initial model uses 24 complete months of observations to predict the following six months for each H3 cell.

The first useful result needs both a fair baseline comparison and evidence that learned embeddings add value beyond the input features.
A negative result remains a valid outcome.

Milestone 0 and [Gate 0](SPEC.md#milestone-0--minimal-scaffold) are complete.
The scaffold uses Python 3.14 and requires the AMD GPU through the main Compose service.
The RX 7900 XTX passed tensor and gradient checks in this service.
A synthetic GRU forward pass, backward pass, and optimizer step also passed on this GPU.
Data preparation and training remain unimplemented.
The next implementation milestone is Milestone 1, subject to the owner's instruction after [Gate 0](SPEC.md#milestone-0--minimal-scaffold).

## Prototype priorities

The temporal learning experiment takes priority over supporting infrastructure.
OSM preparation must supply trustworthy cell-month features through a small, direct implementation.
Files, configuration, and local checkpoints are sufficient for the first experiment.

Inputs must respect prediction cutoffs, and reserved test data must remain separate from development decisions.
Baselines must receive the same permitted information as the neural model.
Geometry omissions must preserve the specified meaning of numeric subset features and unavailable cell-month features.

ML work requires the AMD GPU. Commands must fail if it is unavailable.
Spatial context, continual learning, live updates, and an inspection application remain deferred until the earlier experiment supplies useful evidence.

## Implemented capabilities

- The installed `atlas` CLI reports runtime versions and GPU availability, then checks a small GPU tensor operation and its gradient.
- `atlas doctor` requires AMD execution and rejects a non-ROCm build or an unavailable GPU.
- `uv.lock` fixes the ROCm PyTorch dependencies and the pytest, Ruff, `ty`, and pre-commit tools.
- One Docker image and one Compose service configure the ROCm runtime and WSL GPU access.
- Docker volumes retain the Python environment and dependency cache across container runs and image builds.
- One smoke test checks the installed CLI entry point with `atlas --help`. GPU execution requires a manual Compose check.
- The README provides startup commands, development checks, and OSM attribution and license information.

These capabilities establish runtime readiness. They provide no evidence of predictive skill or useful learned representations.

## Remaining work

- [x] **Milestone 0 — GPU scaffold:** standard Compose image, GPU `atlas doctor`, locked dependencies, development checks, startup instructions, and CLI smoke test.
- [ ] **Milestone 1 — Historical-data slice:** reconstruct a small Norwegian sample, produce the three change families, and check counting semantics with fixtures.
- [ ] **Milestone 2 — Norway corpus:** produce cell-month Parquet data, fixed temporal and geographic splits, development summaries, and a PyTorch data loader.
- [ ] **Milestone 3 — Baselines:** compare zero change, recent-rate persistence, and boosted trees, then set the minimum worthwhile neural improvement.
- [ ] **Milestone 4 — Temporal learner:** resumable GRU training, exported embeddings, model and representation comparisons, repeated seeds, and final test evaluation.

Only the currently approved milestone receives implementation work.
The gates in the specification govern progression through this roadmap.

## Current model evidence

No baseline scores, neural-model results, or embedding comparisons are available.
Predictive skill and useful learned representations remain unproven.

The first comparison needs validation results against the strongest implemented baseline and a linear probe against dimension-matched PCA.
A promising configuration needs two additional seeds.
The final reserved-test evaluation follows the frozen model choice.

## Blockers and next decisions

Docker Desktop can pass `/dev/dxg` into the container. The agent process does not need direct access to this device.
The main Compose service passed GPU runtime checks with Adrenalin 26.8.1, PyTorch 2.14.0, and ROCm 7.2.
The image includes the C++ headers that MIOpen needs to compile GRU kernels at runtime.
The ROCm profiler aborts on WSL because it expects Linux KFD interfaces.
Compose disables profiler registration with `ROCPROFILER_REGISTER_ENABLED=0`; GPU calculations remain enabled.
This small runtime check does not establish training speed or stability during long runs.

The history source, usable years, geometry reconstruction, and Norway-wide resource requirements need evidence from the small data slice at Gate 1.
These later decisions do not block the scaffold.
