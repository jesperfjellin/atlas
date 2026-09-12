# Atlas progress

This is the single product roadmap for the Atlas OSM learning experiment.
The [specification](SPEC.md), version 0.2, defines the experiment, technical contracts, and milestone gates.
This document tracks capabilities, remaining work, model evidence, and blockers.

## Product and current position

Atlas aims to predict future changes in OpenStreetMap and learn useful representations of places through time.
The experiment uses historical OSM data from Norway and runs on one consumer desktop.
The initial model uses 24 complete months of observations to predict the following six months for each H3 cell.

The first useful result needs both a fair baseline comparison and evidence that learned embeddings add value beyond the input features.
A negative result remains a valid outcome.

Milestone 0 is complete: the repository provides a runnable CPU scaffold for Python 3.14.
The `atlas doctor` command works locally and through Docker Compose.
The owner approved the optional RX 7900 XTX compatibility check at [Gate 0](SPEC.md#milestone-0--minimal-scaffold).
GPU execution remains unverified.
Data preparation and training remain unimplemented.
The next implementation milestone is Milestone 1, subject to the owner's instruction after [Gate 0](SPEC.md#milestone-0--minimal-scaffold).

## Prototype priorities

The temporal learning experiment takes priority over supporting infrastructure.
OSM preparation must supply trustworthy cell-month features through a small, direct implementation.
Files, configuration, and local checkpoints are sufficient for the first experiment.

Inputs must respect prediction cutoffs, and reserved test data must remain separate from development decisions.
Baselines must receive the same permitted information as the neural model.
Geometry omissions must preserve the specified meaning of numeric subset features and unavailable cell-month features.

CPU execution is required. AMD acceleration remains optional.
Spatial context, continual learning, live updates, and an inspection application remain deferred until the earlier experiment supplies useful evidence.

## Implemented capabilities

- The installed `atlas` CLI reports runtime versions and GPU availability, then checks a small tensor operation and its gradient.
- `atlas doctor --device rocm` requires AMD execution and rejects a non-ROCm build or an unavailable GPU. CPU remains the default.
- `uv.lock` fixes the CPU PyTorch dependencies and the pytest, Ruff, `ty`, and pre-commit tools.
- One Docker image and one Compose service support the CLI and development checks, with a separate container environment.
- One smoke test checks that the installed CLI runs on CPU. GPU execution requires a manual environment check.
- The README provides startup commands, development checks, and OSM attribution and license information.

These capabilities establish runtime readiness. They provide no evidence of predictive skill or useful learned representations.

## Remaining work

- [x] **Milestone 0 — CPU scaffold:** runnable `atlas doctor`, locked dependencies, development checks, one Compose service, startup instructions, and a smoke test.
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

No technical blocker remains for Milestone 0.
The CPU runtime works locally and through Compose.
The owner confirmed that `/dev/dxg` exists in the regular WSL terminal, but the agent sandbox cannot access this device.
This restriction blocks the approved AMD device check. GPU calculations, gradients, and access through Compose remain unverified.
The [PyTorch ROCm package index](https://download.pytorch.org/whl/rocm7.2/torch/) includes a Python 3.14 build of PyTorch 2.14.0.
Package availability does not establish compatibility with the owner's Windows driver or prove GPU execution.
An AMD Compose configuration awaits the successful device check required by [the runtime contract](SPEC.md#121-docker-compose).
The CPU path is ready for the next milestone.

The history source, usable years, geometry reconstruction, and Norway-wide resource requirements need evidence from the small data slice at Gate 1.
These later decisions do not block the scaffold.
