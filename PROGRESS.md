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

- The installed `atlas` CLI reports runtime versions and GPU availability, then performs a small CPU tensor operation.
- `uv.lock` fixes the CPU PyTorch dependencies and the pytest, Ruff, `ty`, and pre-commit tools.
- One Docker image and one Compose service support the CLI and development checks, with a separate container environment.
- A smoke test exercises the installed CLI and its CPU tensor check.
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
The CPU PyTorch build reports no accessible GPU. This result does not establish whether the machine supports AMD acceleration.
At [Gate 0](SPEC.md#milestone-0--minimal-scaffold), the owner must decide whether to configure optional AMD acceleration now.
The CPU path is ready for the next milestone.

The history source, usable years, geometry reconstruction, and Norway-wide resource requirements need evidence from the small data slice at Gate 1.
These later decisions do not block the scaffold.
