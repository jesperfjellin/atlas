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
Milestone 1 is implemented and demonstrated on the supplied Norway history file. Training remains unimplemented.
The next step is the [Gate 1](SPEC.md#milestone-1--real-historical-data-vertical-slice) decision before a Norway corpus build.
The Kristiansand domain in `configs/kristiansand.toml` was designated development data through December 2023 before inspection, including earlier reconstruction history.
This domain and period must not enter the reserved test partition.

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
- `atlas build-dataset --config` reads a bounded history extract and writes monthly Parquet rows for a fixed H3 cell set.
- Historical reference lookup reconstructs nodes, ways, and complete simple multipolygons at each relevant time.
- Raw edits, semantic additions/removals, mapped state, and net state change remain separate, including child-induced geometry movement.
- Five synthetic tests protect change counting, historical geometry, month boundaries, missing geometry, and Parquet availability semantics.
- The README provides extraction, build, development, and attribution instructions, along with the small taxonomy and geometry limits.

These capabilities establish runtime and input-preparation readiness. They provide no evidence of predictive skill or useful learned representations.

## Remaining work

- [x] **Milestone 0 — GPU scaffold:** standard Compose image, GPU `atlas doctor`, locked dependencies, development checks, startup instructions, and CLI smoke test.
- [x] **Milestone 1 — Historical-data slice:** reconstruct a small Norwegian sample, produce the three change families, and check counting semantics with fixtures.
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

## Historical-data evidence

The Geofabrik Norway history file was reduced to a buffered Kristiansand extract with Osmium's history-aware `complete_ways` strategy.
The 22.2 MiB extract took 17 seconds to create, with about 7.3 GB of peak extraction memory.
The local file retains 1,798,573 entities after the builder discards versions beyond the slice and keeps one pre-start version per entity.

The 2023 build produced 156 rows: 13 fixed H3 resolution-6 cells across 12 complete months, with 51 feature columns.
Outputs are in `data/derived/kristiansand-2023/`. Every row has source coverage for this interval.
The build took 251 seconds and used 2,109 MiB of peak memory.
It attempted 18,520,492 geometry assignments and skipped 12,506.
These counters include objects outside the study cells in the buffered extract; they do not measure feature completeness within the cells.
Net state change and semantic totals differ for land use and other POIs in this slice.
A manual reconciliation attributed those differences entirely to omitted transitions with unreconstructable geometry, consistent with the required subset semantics.

Manual development examples matched the intended semantics:

- Way `1126183431`, version 1, created a 51.3 m path on 2023-01-01: one raw creation and one semantic road-path addition.
- Way `37452395`, versions 15–16, lost `landuse=industrial` on 2023-09-26 while remaining visible: a modification and semantic land-use removal.
- Way `38358690`, versions 11–12, changed geometry on 2023-01-17 while retaining `amenity=parking`: a modification without a category addition or removal.
- Way `4136259`, versions 3–4, became invisible on 2023-09-26: a deletion assigned to its previous cell.

## Blockers and next decisions

Docker Desktop can pass `/dev/dxg` into the container. The agent process does not need direct access to this device.
The main Compose service passed GPU runtime checks with Adrenalin 26.8.1, PyTorch 2.14.0, and ROCm 7.2.
The image includes the C++ headers that MIOpen needs to compile GRU kernels at runtime.
The ROCm profiler aborts on WSL because it expects Linux KFD interfaces.
Compose disables profiler registration with `ROCPROFILER_REGISTER_ENABLED=0`; GPU calculations remain enabled.
This small runtime check does not establish training speed or stability during long runs.

Gate 1 can retain Geofabrik full history, Osmium extraction, and the direct pyosmium reader.
The source contains versions from 2006 through August 2026, but complete early history is not guaranteed before the 2007 API transition.
The configuration conservatively declares coverage from 2008 and this run demonstrates the 2023 slice only.
The file lacks a replication timestamp header, so source coverage remains an explicit configuration declaration.

The small taxonomy and primary-cell implementation are documented in the README and follow [Section 7](SPEC.md#7-exact-meaning-of-change).
Incomplete relations, nested relation members, nested same-role rings, and invalid geometries are omitted under the specified subset semantics.

The current in-memory reader must not receive the full 2.1 GiB Norway file.
A rough linear estimate from compressed size suggests hundreds of GiB of memory and several hours per year for an unpartitioned build.
This estimate is coarse: geometry complexity and compression vary by region, and extraction adds references outside the requested area.
The measured small build fits the Docker daemon's roughly 15.2 GiB memory limit. Norway processing needs bounded geographic chunks in Milestone 2.
Before a Norway run, approve that milestone and create its temporal/geographic split without inspecting reserved-test targets.
