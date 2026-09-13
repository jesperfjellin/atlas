# Atlas progress

This is the single product roadmap for the Atlas OSM learning experiment.
The [specification](SPEC.md), version 0.4, defines the experiment, technical contracts, and milestone gates.
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
Milestones 1 and 2 are complete, including the full Norway corpus and [Gate 2 decisions](SPEC.md#gate-2-decisions--frozen-2026-09-13).
The real Norway sample loader passed GPU acceptance.
Milestone 3 and [Gate 3](SPEC.md#gate-3-decisions--frozen-2026-09-13) are complete. Boosted trees are the strongest baseline on both validation groups and primary metrics.
Milestone 4 awaits approval. Neural training remains unimplemented.
The Kristiansand domain in `configs/kristiansand.toml` was designated development data through December 2023 before inspection, including earlier reconstruction history.
The frozen split excludes all listed Kristiansand development cells from reserved testing at every date.

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
- Synthetic tests protect change counting, historical geometry, month boundaries, missing geometry, and Parquet availability semantics.
- A bounded corpus builder streams entities and stores compact node/way reference arrays. Shared references supply geometry without duplicate feature contributions.
- Completed batches survive interruption. The Norway resume preserved all 951 completed batches; development builds reproduced the original 2023 features.
- `configs/split.yaml` fixes the domain, temporal partitions, geographic holdouts, neighbouring-cell buffers, and Kristiansand exclusion before new target summaries.
- A PyTorch dataset supplies 24-month inputs, six-month change targets, calendar encodings, and availability masks. Synthetic checks protect alignment and training-only input scaling.
- The completed Norway corpus contains 132 monthly Parquet files with 1,906,872 unique cell-month rows and consistent feature availability.
- A real Norway batch passed GPU loading on the RX 7900 XTX: inputs `(32, 24, 104)`, targets `(32, 6, 37)`, and boolean target masks.
- Saved input preprocessing uses unique training input cell-months only. Development summaries cover training and both validation groups, without reserved-test target summaries.
- The frozen scorer preserves target masks, tied occurrence scores, and equal weights for target families and forecast months.
- `atlas train-baselines --config` compares zero change, a six-month recent rate, and separate LightGBM models for all 222 target/horizon outputs.
- LightGBM bins flattened inputs in batches and reuses those bins. Completed models survive interruption. Training and forecast execution use the Radeon GPU.
- The complete OpenCL run fitted 222 target/horizon models using all 768,423 training samples and scored both natural validation groups.
- Resuming the completed run reused all 222 saved models without changing their files and reproduced the aggregate validation scores.
- A small checkpoint check rejects tree updates outside the bounds permitted by the squared loss and observed training labels.
- The README explains the experiment, its intended evidence, and data attribution. The specification records the feature and geometry conventions.

The baseline results demonstrate predictable OSM activity under the frozen validation split.
Neural forecasting gains and useful learned representations remain unproven.

## Remaining work

- [x] **Milestone 0 — GPU scaffold:** standard Compose image, GPU `atlas doctor`, locked dependencies, development checks, startup instructions, and CLI smoke test.
- [x] **Milestone 1 — Historical-data slice:** reconstruct a small Norwegian sample, produce the three change families, and check counting semantics with fixtures.
- [x] **Milestone 2 — Norway corpus:** cell-month Parquet data, fixed temporal and geographic splits, development summaries, a GPU-verified PyTorch loader, leakage checks, and frozen Gate 2 decisions.
- [x] **Milestone 3 — Baselines:** zero change, recent-rate persistence, and boosted trees evaluated; strongest baseline identified and minimum worthwhile neural improvement frozen at Gate 3.
- [ ] **Milestone 4 — Temporal learner:** resumable GRU training, exported embeddings, model and representation comparisons, repeated seeds, and final test evaluation.

Only the currently approved milestone receives implementation work.
The gates in the specification govern progression through this roadmap.

## Current model evidence

All three baselines have been evaluated on all 217,911 temporal-validation samples and 18,620 geographic-validation samples.
Scores use the frozen signed-log RMSE and occurrence average precision:

| Validation | Baseline | Signed-log RMSE | Average precision |
| --- | --- | ---: | ---: |
| Temporal | Zero change | 0.824629 | 0.057871 |
| Temporal | Recent rate | 1.289597 | 0.162328 |
| Temporal | Boosted trees | **0.733758** | **0.307705** |
| Geographic | Zero change | 0.787893 | 0.060463 |
| Geographic | Recent rate | 1.219025 | 0.183421 |
| Geographic | Boosted trees | **0.690916** | **0.342283** |

Recent activity improves occurrence ranking, but its arithmetic rate forecasts have larger magnitude errors than zero change.
Boosted trees reduce aggregate magnitude error by 11.0% on temporal validation and 12.3% on geographic validation relative to zero change.
Their occurrence AP is 89.6% and 86.6% higher, respectively, than recent-rate persistence.
This supports trying the compact GRU under the [frozen Gate 3 criterion](SPEC.md#gate-3-decisions--frozen-2026-09-13).

The reference run is `runs/baselines-norway-opencl/`, with resolved configuration, logs, 222 saved model files, and complete `metrics.json`.
Metrics include all six horizons, three target families, and per-target observation and positive supports for each baseline and validation group.
Every checkpoint passed squared-loss update bounds and the configured depth limit.
Five full-data acceptance fits also matched native LightGBM validation losses with the GPU evaluator, including repeated rare-count fits and signed net changes.
The full saved-model resume reproduced the aggregate scores.

The earlier native-ROCm tree models and scores in `runs/baselines-norway/` are invalid: fitting produced updates outside the bounds allowed by the loss and training labels.
The replacement uses OpenCL with double-precision histograms. It reuses only the earlier run's valid input-bin files through symlinks; keep both directories.
All replacement tree checkpoints were fitted anew.

This evidence concerns aggregate mapping activity within H3 resolution-6 areas. It does not locate individual changes within those areas or distinguish construction from later mapping.
No neural-model results or embedding comparisons are available. Milestone 4 must compare its representations with dimension-matched PCA as well as meeting the forecast criterion.
The final reserved-test evaluation follows the frozen model choice; reserved-test targets have not been inspected during baseline development.

## Norway corpus evidence

The full corpus is in `data/derived/norway-2015-2025/`.
It contains 14,446 fixed H3 resolution-6 cells across 132 complete months from January 2015 through December 2025.
All 1,906,872 rows passed date, cell, schema, dtype, availability, and finite-value checks.
Each month contains every fixed cell, including empty cells and geographic buffers.
Source coverage is declared for every month; individual geometry omissions retain numeric subset features under [Section 7](SPEC.md#7-exact-meaning-of-change).

The build processed 236,469,054 nodes, 13,431,132 ways, and 889,140 relations.
It took about 8 hours 26 minutes of wall time, including one corrective resume, and reached 4,153 MiB of peak process memory.
Outputs and reference arrays occupy 15.27 GiB; the monthly Parquet files themselves use about 54 MiB.
The two geometry counters are 306,205,228 attempted and 1,548,549 skipped assignments.
These counters include candidates outside the study cells and do not measure cell-level completeness.
Some polygon assembly calls emitted numerical warnings. The resulting corpus passed all finite-value and structural checks.

The fixed domain uses the Natural Earth land boundary for mainland Norway, Svalbard, and Jan Mayen.
Geographic groups contain 11,469 training cells, 980 validation cells, 1,002 reserved-test cells, and 995 buffer cells.
Every previously inspected Kristiansand cell remains excluded from reserved testing.
Actual loader counts are:

| Partition | Samples |
| --- | ---: |
| Training | 768,423 |
| Temporal validation | 217,911 |
| Geographic validation | 18,620 |
| Reserved temporal test | 80,234 |
| Reserved geographic test | 7,014 |

The reserved-test checks covered only structure, dates, eligibility, and tensor loading.
They did not inspect target distributions, predictions, or model metrics.
`preprocessing.npz` contains the saved training-input statistics. `development-summary.json` contains only training and validation target summaries, plus structural sample counts.

Development summaries contain 825,768 training cell-months, 275,256 temporal-validation cell-months, and 23,520 geographic-validation cell-months.
Selected zero-value percentages show the natural sparsity:

| Target | Training | Temporal validation | Geographic validation |
| --- | ---: | ---: | ---: |
| Raw creations | 83.34% | 86.32% | 85.20% |
| Raw modifications | 80.52% | 81.04% | 79.15% |
| Building additions | 96.93% | 98.77% | 98.72% |
| Road-path additions | 96.18% | 96.84% | 96.30% |
| Net building area | 96.50% | 98.18% | 98.11% |
| Net road length | 92.71% | 92.95% | 91.97% |

Training building additions average 3.158 per cell-month, compared with 0.062 in temporal validation.
Rare training bursts reach 165,570 raw creations in a cell-month; net land-use area changes reach about 875 million square metres in magnitude.
Whole geometries are assigned to one primary cell, so an assigned area can exceed the cell's area.
Area and length changes also contain sub-unit values. The occurrence definition excludes those values, while magnitude scoring retains them.

This evidence supports retaining the small taxonomy, using signed-log targets with squared-error loss, and measuring both magnitude and occurrence.
The exact transforms, weighting, and primary metrics are frozen in [Gate 2](SPEC.md#gate-2-decisions--frozen-2026-09-13).

The Geofabrik source contains versions from April 2005 through August 2026 and lacks a replication timestamp header.
Coverage therefore remains an explicit declaration in `configs/norway.toml`, starting in 2008; the completed experiment uses only 2015–2025.
The reader retains the latest version before the build start and rejects timestamp reversals that affect the retained timeline.

## Development-slice evidence

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

The bounded builder matched all 156 original rows and all 51 features.
The 2015–2025 development pilot produced 1,716 rows, preserved 22 completed batches on restart, and again matched the original 2023 features.

Manual development examples matched the intended semantics:

- Way `1126183431`, version 1, created a 51.3 m path on 2023-01-01: one raw creation and one semantic road-path addition.
- Way `37452395`, versions 15–16, lost `landuse=industrial` on 2023-09-26 while remaining visible: a modification and semantic land-use removal.
- Way `38358690`, versions 11–12, changed geometry on 2023-01-17 while retaining `amenity=parking`: a modification without a category addition or removal.
- Way `4136259`, versions 3–4, became invisible on 2023-09-26: a deletion assigned to its previous cell.

## Blockers and next decisions

There is no remaining Milestone 3 blocker. The next decision is approval of [Milestone 4](SPEC.md#milestone-4--first-temporal-learner).
The baseline evidence supports that experiment, and Gate 3 fixes its forecast success criterion before neural training.

The OpenCL GPU baseline run is complete. A small GRU optimizer step has passed, but full neural training speed, stability, and predictive value remain unmeasured.
Geometry omissions, the approximate study boundary, and coarse cell-level aggregation remain limitations of the fixed experiment.
Reserved-test targets remain unavailable for development decisions until the final model choice is frozen.
