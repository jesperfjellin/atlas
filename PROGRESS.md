# Atlas progress

This is the single product roadmap for the Atlas OSM learning experiment.
The [specification](SPEC.md), version 0.6, defines the experiment, technical contracts, and milestone gates.
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
Milestone 3 and [Gate 3](SPEC.md#gate-3-decisions--frozen-2026-09-13) are complete. Boosted trees won the original baseline comparison and remain the frozen Gate 3 reference.
Milestone 4 is approved and in progress. Training, resume, validation exports, and embedding checks are implemented.
Two complete GRU development runs trail the tree baseline; their linear embedding probes also trail dimension-matched PCA.
The [baseline-and-diagnosis campaign](#baseline-and-diagnosis-campaign) is complete. A linear state-and-activity-summary model nearly matches tree magnitude errors and improves occurrence ranking.
All four paired residual-MLP runs completed their 120-epoch schedules. None beats trees; their best checkpoints all come from epoch 2.
The selected MLP's embedding probe improves occurrence ranking over PCA but worsens magnitude error. The capacity study is complete.
The neural model choice remains open, and reserved-test targets remain closed.
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
- `atlas train --config` fits the compact GRU or approved residual MLPs on shuffled training windows and selects the best epoch using full temporal-validation magnitude error.
- Neural training saves atomic `latest.pt` and `best.pt` checkpoints, including optimizer state, epoch, step, early stopping state, configuration, and random state. `--resume` continues from `latest.pt`.
- The learner exports validation embeddings and six-month forecasts as Parquet, keyed by cell and cutoff, with the frozen target order and signed-log units recorded.
- Short GPU acceptance trained on 4,096 real examples and exported 304 development predictions and embeddings. A resumed optimizer update matched uninterrupted training; validation RMSE matched the frozen scorer.
- Two full GRU runs used all 768,423 training samples and exported predictions and embeddings for all 236,531 validation samples.
- Resuming the completed initial GRU run preserved both checkpoints and reproduced every saved validation metric.
- Residual MLPs support summary inputs or summaries plus ordered history, two declared capacity pairs, dropout, and an epoch-based learning-rate schedule.
- MLP runs prepare fixed inputs on the GPU once. Additional summary scaling uses observed training inputs only and is saved with the model.
- All four MLP shapes passed real-data GPU acceptance. Restored dropout, optimizer state, and the learning-rate reduction reproduced the next update exactly; cached validation matched the frozen scorer.
- Four complete 120-epoch MLP runs exported both full validation groups. Resuming the selected completed run preserved checkpoints, training logs, and runtime measurements while reproducing all metrics.
- `atlas explore-embeddings --run` compares neural and dimension-matched PCA representations with ridge probes, same-cutoff nearest neighbours, and a few cell trajectories.
- PCA, probe scaling, and probe fitting use the same fixed, target-independent training subset. Both complete validation groups remain unresampled.
- `atlas diagnose-baselines --config configs/diagnosis.toml` runs the completed full-data ridge/PCA campaign with independent historical preprocessing and five declared penalties per control.
- The campaign saves 89 keyed validation forecast files, aggregate/family/horizon/year scores, geographic-parent scores with supports, and unique cell-month building summaries.
- Full command resume preserves completed fits, forecasts, and scores. The campaign loader stops before the reserved 2025 months.
- The README explains the experiment, its intended evidence, and data attribution. The specification records the feature and geometry conventions.

The baseline results demonstrate predictable OSM activity under the frozen validation split.
Neural forecast gains over trees remain unproven. The selected MLP representation has mixed probe results: better occurrence ranking and worse magnitude error than PCA.

## Remaining work

- [x] **Milestone 0 — GPU scaffold:** standard Compose image, GPU `atlas doctor`, locked dependencies, development checks, startup instructions, and CLI smoke test.
- [x] **Milestone 1 — Historical-data slice:** reconstruct a small Norwegian sample, produce the three change families, and check counting semantics with fixtures.
- [x] **Milestone 2 — Norway corpus:** cell-month Parquet data, fixed temporal and geographic splits, development summaries, a GPU-verified PyTorch loader, leakage checks, and frozen Gate 2 decisions.
- [x] **Milestone 3 — Baselines:** zero change, recent-rate persistence, and boosted trees evaluated; strongest baseline identified and minimum worthwhile neural improvement frozen at Gate 3.
- [ ] **Milestone 4 — Temporal learner:** training, resume, exports, model/representation comparisons, the baseline campaign, and the paired nonlinear capacity study are complete. The next experiment needs a scope decision. Final model selection, conditional seed repeats, and reserved-test evaluation after a frozen final choice remain.

Only the currently approved milestone receives implementation work.
The gates in the specification govern progression through this roadmap.

## Baseline-and-diagnosis campaign

**Complete.** Artifacts are in `runs/diagnosis-norway/`, configured by [configs/diagnosis.toml](configs/diagnosis.toml).
The comparison measures simple learned forecasts, the value of historical inputs, and sensitivity to development periods.
It supports the next [Milestone 4](SPEC.md#milestone-4--first-temporal-learner) decision without identifying a single cause of the GRU results.

### Comparisons

- Six-month recent means computed in signed-log space, with available-month denominators and signed net changes.
- Ridge on all 2,496 ordered input values.
- Full-data, 64-dimensional PCA followed by ridge.
- Ridge on final mapped state, calendar, and availability: 30 inputs.
- Ridge on those inputs plus six- and 24-month transformed change means, active-month fractions, and observed-month fractions: 252 inputs.

Every learned comparator uses all eligible training windows and the same targets and evaluation populations.
The five penalties were fixed before fitting: `0.001`, `0.01`, `0.1`, `1`, and `10`.
Intercepts are unpenalized; penalties are relative to observed sample counts.
Representation scaling uses training windows only. PCA uses the exact centered training covariance and standardizes its 64 components for ridge.
Empty lookbacks produce zero means and active fractions, accompanied by zero observed fractions.

### Historical comparison and safeguards

The three folds train on 2017–2019, 2017–2020, and 2017–2021 targets, respectively.
They contain 355,539, 493,167, and 630,795 training windows, with 80,283 validation windows in each following year.
Each six-month target window remains inside its partition. All geographic exclusions and buffers remain fixed.
Each fold independently fits normalization on unique training input cell-months, ending in June 2019, June 2020, or June 2021.
PCA, representation scaling, and regression also use only that fold's training data.

All 60 historical ridge comparisons finished. Mean aggregate RMSE across the three folds selected these penalties:

| Control | Penalty | 2020 RMSE | 2021 RMSE | 2022 RMSE | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordered history | 10 | 0.820658 | 0.960482 | 0.710981 | 0.830707 |
| PCA, 64 dimensions | 0.1 | 0.820649 | 0.965428 | 0.710210 | 0.832096 |
| State and calendar | 0.1 | 0.825228 | 0.964013 | 0.714721 | 0.834654 |
| State and activity summaries | 1 | 0.818841 | 0.962777 | 0.708791 | 0.830137 |

Ordered ridge prefers the largest declared penalty in every fold. This bounded search does not locate an unrestricted optimum.
Absolute error varies considerably by year, including for zero change: its fold RMSEs are 0.907043, 1.087551, and 0.841279.
These are descriptive period differences, not proof that recent-only training would help.

### Original validation results

Each selected control was refitted on all 768,423 original training windows.
Scores cover all 217,911 temporal and 18,620 geographic validation windows, using the [frozen scorer](SPEC.md#92-metrics).
Original split and preprocessing files remain intact. Saved trees and GRUs were evaluated only on their original validation period.

| Comparator | Temporal RMSE | Temporal AP | Geographic RMSE | Geographic AP |
| --- | ---: | ---: | ---: | ---: |
| Recent mean in signed-log space | 0.780597 | 0.211674 | 0.730211 | 0.242468 |
| Full-data ordered ridge | 0.739631 | **0.321851** | 0.707277 | **0.350471** |
| Full-data PCA/ridge | 0.735342 | 0.316011 | 0.698563 | 0.343317 |
| State/calendar ridge | 0.740981 | 0.310974 | 0.708802 | 0.345536 |
| State/summary ridge | **0.732522** | 0.319960 | 0.693459 | 0.348504 |
| Frozen tree reference | 0.733758 | 0.307705 | **0.690916** | 0.342283 |

Computing recent means in transformed space reduces RMSE by 39.5% and 40.1% relative to the original arithmetic-rate baseline.
It also beats zero change, but remains behind the learned controls.
State/summary ridge has 0.17% lower temporal RMSE and 0.37% higher geographic RMSE than trees.
Its AP is higher by 0.012255 and 0.006221, respectively. These AP differences are not percentages of correct predictions.
Simple learned statistics therefore account for nearly all demonstrated forecast performance. The existing GRUs add no measured advantage.

Ordered ridge improves RMSE over state/calendar alone by only 0.18% temporally and 0.22% geographically.
Activity summaries improve it by 1.14% and 2.16%, respectively, and outperform ordered ridge in both original validation groups.
This establishes useful additional history in the summary pipeline. It does not establish which temporal patterns matter or rule out nonlinear gains from ordered history.

For state/summary ridge, temporal family RMSEs are 1.065598 raw, 0.139774 semantic, and 0.674337 net.
Geographic family RMSEs are 1.008235, 0.141325, and 0.637297.
Only its temporal raw-family magnitude error beats trees; its occurrence AP exceeds trees in all six family/group comparisons.
It slightly beats tree RMSE at all six temporal horizons. Geographically it is close at horizons 1–3 and trails at horizons 4–6.
All family, horizon, and per-target scores and supports remain in the saved metric files.

### Period and geographic diagnosis

Year scores use each forecast's actual target year, including windows crossing December.
State/summary RMSE is 0.755237 and 0.712823 for temporal targets in 2023 and 2024; tree scores are 0.754867 and 0.715929.
Geographic state/summary scores are 0.646966 and 0.727070; tree scores are 0.636011 and 0.731554.
The year-to-year direction therefore differs across geographies.
Overlapping windows are dependent, and per-year horizon supports differ. These comparisons are not independent-sample significance tests.

Eight designated geographic parents have eligible validation cells, with 152–4,940 windows per parent.
Parent `830705fffffffff` has one study cell, excluded by the frozen buffer rule; its scores are explicitly unavailable with zero support.
Across the other parents, state/summary RMSE ranges from 0.126312 to 0.969085, versus 0.160529 to 0.961130 for trees.
The apparent 21.3% improvement in the smallest parent uses only eight cells and 152 windows; zero change does better there than either learned model.
In the other seven parents, state/summary RMSE differs from trees by less than 0.83% in either direction.
Zero change beats trees on four of the eight nonempty parents. Overall success does not imply useful magnitude forecasts in every area.
All nine parent entries include per-target observation and positive supports. Pooled AP includes ranking between parents and is not their average local AP.

Building additions use unique development cell-months, without window duplication:

| Training-geography year | Positive months | Mean when positive | Median when positive | Additions in top 1% of positive months |
| --- | ---: | ---: | ---: | ---: |
| 2017 | 1.38% | 21.43 | 3 | 34.88% |
| 2018 | 3.38% | 30.33 | 2 | 45.42% |
| 2019 | 2.42% | 54.23 | 5 | 33.53% |
| 2020 | 2.30% | 30.96 | 4 | 33.62% |
| 2021 | 7.32% | 192.45 | 33 | 21.65% |
| 2022 | 1.62% | 93.54 | 3 | 39.09% |
| 2023 | 1.11% | 4.71 | 2 | 16.31% |
| 2024 | 1.35% | 5.35 | 2 | 16.17% |

Each row covers 137,628 observed cell-months. The top-1% count rounds upward to a whole positive month.
Geographic validation has 11,760 cell-months per year: positive fractions 1.18%/1.38%, positive means 4.53/4.48, and medians 2/2.
Its top-1% shares are 14.13%/27.96%; the latter contains just two positive cell-months.
The 2021 shift appears in both frequency and the median positive size, not just a few extreme values.
The data establishes a change in observed OSM activity; it does not attribute it to a particular import or establish the best adaptation strategy.

### Completion and next experiment

The full computational pass took 8 minutes 40 seconds on the RX 7900 XTX.
Peak process memory was 3.45 GiB; GPU allocations peaked at 2.47 GiB, with 4.09 GiB reserved by the allocator.
Float64 GPU regression agreed with independent small reference solves within `2e-12`; PCA variances matched an independent SVD.
The run directories retain configurations, all preprocessing and fits, 89 keyed forecasts, metrics, and building summaries.
All declared comparisons completed. Full command resume preserves fitted models, forecasts, and scores.

**Completed follow-up:** the paired study below compared a nonlinear network using state/activity summaries with one receiving those same summaries plus ordered monthly history.
The hypothesis is that nonlinear relationships can improve the strong linear forecast; the paired history control tests whether detailed monthly inputs add further value.
Use residual MLPs with a 64-dimensional embedding, two declared capacity budgets, and approximately matched parameter counts within each pair.
Keep the optimizer, regularization, learning-rate schedule, training population, validation rule, and maximum training budget matched.
Allow training to continue through scheduled learning-rate reductions before declaring a plateau; record learning curves, parameter counts, time, and memory.
This starts a controlled neural capacity study with a declared comparison and training budget.
MLP temporal encoders have precedent in [TiDE](https://arxiv.org/abs/2304.08424), but its benchmark results do not establish an Atlas improvement.

Gate 3, the loss, and the reserved-test restriction remain unchanged. The paired MLP study below is complete; recency weighting, occurrence heads, and new losses remain outside its scope.
The baseline campaign does not complete Milestone 4.
Spatial context, finer cells, source-history rebuilds, and reserved-test evaluation remain outside its scope.

## Paired nonlinear capacity study

**Complete.** The following design was fixed before training:

| Inputs | Width | Trainable parameters | Capacity pair |
| --- | ---: | ---: | --- |
| [State and activity summaries, 252 values](configs/mlp-summary-small.toml) | 205 | 249,629 | Small |
| [Same summaries plus ordered history, 2,718 distinct values](configs/mlp-history-small.toml) | 76 | 249,866 | Small |
| [State and activity summaries, 252 values](configs/mlp-summary-large.toml) | 457 | 999,329 | Large |
| [Same summaries plus ordered history, 2,718 distinct values](configs/mlp-history-large.toml) | 258 | 1,001,344 | Large |

Each model has an input projection, two pre-normalized residual MLP blocks, a 64-value embedding, and a linear 222-output forecast head.
Each block uses two same-width linear layers, GELU, LayerNorm, and dropout 0.1.
Widths differ to match parameter budgets; the comparison does not isolate temporal order from every architectural effect.
Keep the original numeric input normalization and unscaled calendar/availability channels.
Standardize engineered change means using observed training-window summaries only; leave activity and observed fractions unscaled.

Use all 768,423 unresampled training windows, batch size 512, AdamW, learning rate 0.0003, weight decay 0.01, and gradient clipping at 1.
Run every configuration for 120 epochs with no early stopping. Multiply the learning rate by 0.3 after epochs 30, 60, and 90.
Save the best checkpoint by full temporal-validation signed-log RMSE; score both full validation groups afterward.
Use seed 20260913 and the same per-epoch sample permutation across runs.
Checkpoints must retain optimizer, epoch, random state, and the declared schedule so interruption does not reset training.
Measure parameter counts, learning curves, runtime, and process/GPU memory. Verify numerical behavior and resume before full runs.

Choose the candidate for the existing embedding checks by the lowest mean RMSE relative to trees across the two validation groups; ties follow table order.
If a configuration meets both frozen Gate 3 criteria in both groups, repeat it with two additional seeds before claiming a worthwhile gain.
Otherwise, report all four results without seed repeats. Keep reserved-test targets closed and end at a measured recommendation.
No broader architecture search, new loss, spatial context, or source-data rebuild is part of this study.

### Forecast results and training budget

All runs completed 120 epochs on all 768,423 training windows. Each selected epoch 2 using full temporal-validation RMSE.
Each exported all 217,911 temporal-validation and 18,620 geographic-validation forecasts and 64-value embeddings.

| Model | Temporal RMSE | Temporal AP | Geographic RMSE | Geographic AP |
| --- | ---: | ---: | ---: | ---: |
| Summaries, 250k parameters | 0.734177 | 0.302224 | 0.699831 | 0.328964 |
| History and summaries, 250k | 0.739253 | 0.299104 | 0.706725 | 0.330657 |
| Summaries, 1M | 0.735676 | 0.298990 | 0.702490 | 0.320023 |
| History and summaries, 1M | 0.740521 | 0.291422 | 0.708540 | 0.318155 |
| Frozen tree reference | **0.733758** | **0.307705** | **0.690916** | **0.342283** |

The small summary model has the lowest mean relative RMSE and is the declared embedding candidate.
Its magnitude errors are 0.057% and 1.290% worse than trees, respectively; its occurrence AP is also lower in both groups.
The full-data linear summary control remains stronger on both metrics in both groups.
Adding detailed history worsens selected-checkpoint magnitude error at both parameter budgets.
Quadrupling the approximate parameter budget also worsens magnitude error and occurrence AP for both input choices.
These are comparisons of the declared configurations at one seed, with different widths within each capacity pair.
They do not show that temporal ordering is intrinsically useless or establish a ceiling for other neural configurations.

No run meets Gate 3 in either validation group. The declared condition for additional seeds is therefore unmet, and no seed repeats were run.
All three target families have worse geographic magnitude error than trees for every MLP.
The small summary model has slightly better temporal raw-edit RMSE and raw-edit AP in both groups, but worse semantic and net-family scores.
There is no broad semantic-family gain hidden by the aggregate gate.

Longer training reduced training loss while validation error rose. Minimum temporal RMSE in each scheduled phase was:

| Model | Epochs 1–30 | 31–60 | 61–90 | 91–120 |
| --- | ---: | ---: | ---: | ---: |
| Summaries, 250k | 0.734177 | 0.767693 | 0.774647 | 0.777977 |
| History and summaries, 250k | 0.739253 | 0.759263 | 0.768369 | 0.771289 |
| Summaries, 1M | 0.735676 | 0.799739 | 0.818230 | 0.825723 |
| History and summaries, 1M | 0.740521 | 0.790836 | 0.816356 | 0.823790 |

None of the three learning-rate reductions recovered an improvement over the early checkpoint.
The larger models fit training data more closely and have worse late validation error.
This supports early overfitting under the current split and training setup; it does not identify its cause.
The experiment directly tests extended training for these configurations, without demonstrating that every training strategy would behave similarly.

The four runs took 15.36, 14.78, 16.43, and 16.91 minutes, respectively: 63.49 minutes total, including input preparation and validation exports.
GPU allocations peaked at 2.25–2.27 GiB for summary models and 11.49–11.51 GiB for history models; maximum allocator reservation was 11.74 GiB.
Peak process memory across runs was 3.66 GiB. All runs fit the workstation without a memory or numerical failure.
Artifacts are in the four `runs/mlp-{summary,history}-{small,large}-seed20260913/` directories named by the linked configurations.
Each retains configuration, complete epoch logs, best/latest checkpoints, full family/horizon/target metrics, keyed forecasts, embeddings, and runtime measurements.
The combined learning curves are `runs/mlp-capacity-curves.png` and `.svg`.

### Embedding evidence and next experiment

The selected small summary MLP uses the existing 64-dimensional PCA/probe comparison on the same 252 prepared inputs.
PCA, probe scaling, and both probe fits use the declared 32,768 training windows. The neural encoder itself learned from all training windows.
The comparison retains ridge strength 0.01 and both complete validation groups:

| Validation | Probe representation | Signed-log RMSE | Average precision |
| --- | --- | ---: | ---: |
| Temporal | MLP embedding | 0.738198 | **0.308215** |
| Temporal | PCA | **0.736164** | 0.295931 |
| Geographic | MLP embedding | 0.709648 | **0.332274** |
| Geographic | PCA | **0.701431** | 0.323839 |

The MLP probe improves AP by 4.15% and 2.60% relative to PCA, while increasing RMSE by 0.28% and 1.17%.
This is a limited positive result for occurrence ranking in the learned representation.
It is a single-seed, supervised-target probe with subset-fitted comparators; it does not establish independent transfer or satisfy the forecast gate.
The PCA inputs differ from the earlier GRU check, so those PCA scores are not interchangeable.

Inspected June 2024 neighbours show understandable mapped-density groupings for both representations.
The Kristiansand MLP query has 12,712 mapped buildings; its five neighbours have 9,994–17,096.
The Oslo query has 20,776 buildings and retrieves another densely mapped cell near Trondheim among its neighbours.
The eligible query north of Tromsø has 42 buildings and zero June edits; its MLP neighbours have 34–125 buildings, and four also have zero June edits.
These descriptions use input-month features only. Density similarities and high cosine scores do not establish useful future-change predictions.
The three trajectory plots show different place-time paths on training-fitted PCA axes; no causal interpretation is assigned to their direction or distance.
The selected run retains `embedding-checks.json`, fitted PCA/probe tensors, and `trajectories.png`/`.svg`.

**Recommendation, requiring a scope decision:** test recency weighting against uniform weighting using the existing summary inputs, linear control, and small MLP.
Use earlier development folds to select a bounded weighting comparison, with each fold's own training-only preprocessing and complete target windows.
This directly tests whether older training activity reduces relevance to later periods while retaining the current objectives and Gate 3 criterion.
The observed activity shift and learning curves motivate this hypothesis; neither establishes its cause or guarantees a recency benefit.
Further capacity or longer training alone is a lower priority after this completed comparison.
Other architectures, regularization choices, and objectives remain possible future experiments. Reserved-test evaluation remains closed until a final model choice is frozen.

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
These results motivated the compact GRU comparison under the [frozen Gate 3 criterion](SPEC.md#gate-3-decisions--frozen-2026-09-13).

Calculated from the temporal tree scores, semantic targets contribute 1.1988% of aggregate squared error.
Perfect semantic predictions with raw and net errors unchanged would reduce aggregate RMSE by only 0.6012%.
This counterfactual applies to semantic additions/removals; building and road changes also occur in the net family.
Family-level improvements therefore need separate interpretation alongside the frozen aggregate gate.
Occurrence AP ranks point-forecast magnitude; it does not measure calibrated change probabilities, particularly for signed net targets.

The reference run is `runs/baselines-norway-opencl/`, with resolved configuration, logs, 222 saved model files, and complete `metrics.json`.
Metrics include all six horizons, three target families, and per-target observation and positive supports for each baseline and validation group.
Every checkpoint passed squared-loss update bounds and the configured depth limit.
Five full-data acceptance fits also matched native LightGBM validation losses with the GPU evaluator, including repeated rare-count fits and signed net changes.
The full saved-model resume reproduced the aggregate scores.

The earlier native-ROCm tree models and scores in `runs/baselines-norway/` are invalid: fitting produced updates outside the bounds allowed by the loss and training labels.
The replacement uses OpenCL with double-precision histograms. It reuses only the earlier run's valid input-bin files through symlinks; keep both directories.
All replacement tree checkpoints were fitted anew.

This evidence concerns aggregate mapping activity within H3 resolution-6 areas. It does not locate individual changes within those areas or distinguish construction from later mapping.
The GRU uses one layer, 64 embedding values, and 47,070 parameters.
Both exploratory runs use seed `20260913`, all training windows, and early stopping on full temporal-validation magnitude error:

| Validation | GRU configuration | Signed-log RMSE | Average precision |
| --- | --- | ---: | ---: |
| Temporal | Initial | 0.741980 | 0.295818 |
| Geographic | Initial | 0.713247 | 0.337843 |
| Temporal | Lower learning rate, stronger weight decay | 0.740873 | 0.283862 |
| Geographic | Lower learning rate, stronger weight decay | 0.709082 | 0.313142 |

The initial run, [configs/gru.toml](configs/gru.toml), selected epoch 2 and stopped after epoch 8.
The follow-up, [configs/gru-regularized.toml](configs/gru-regularized.toml), selected epoch 1 and stopped after epoch 9.
It changed the learning rate from `0.001` to `0.0003` and weight decay from `0.0001` to `0.01`, retaining the architecture.
Training loss continued to fall while validation error rose. These curves suggest early overfitting under the frozen split.
Neither run beats the reference trees on either primary metric in either validation group, so neither meets Gate 3.
The follow-up slightly reduces magnitude error relative to the initial GRU but worsens occurrence ranking.
No additional seeds have been run because neither configuration is promising under the frozen criterion.

Artifacts are in `runs/gru-norway-64-seed20260913/` and `runs/gru-norway-64-regularized-seed20260913/`.
Each directory contains configuration, epoch logs, latest/best checkpoints, full validation metrics, and keyed Parquet forecasts with embeddings.
Epochs took roughly 27–30 seconds after the first epoch on the RX 7900 XTX.

The [embedding check](SPEC.md#94-embedding-evaluation) uses the same 32,768 training windows for both 64-dimensional representations and their ridge probes:

| Validation | Probe representation | Signed-log RMSE | Average precision |
| --- | --- | ---: | ---: |
| Temporal | Initial GRU | 0.746221 | 0.284587 |
| Temporal | Follow-up GRU | 0.746059 | 0.289765 |
| Temporal | PCA | **0.737280** | **0.306764** |
| Geographic | Initial GRU | 0.720791 | 0.318198 |
| Geographic | Follow-up GRU | 0.717890 | 0.327612 |
| Geographic | PCA | **0.703244** | **0.336576** |

PCA wins this bounded probe comparison. Useful added representation structure is not demonstrated.
That subset comparison did not measure full-data linear performance; the completed campaign above now supplies those controls.
The two short GRU runs use one architecture and seed, change learning rate and weight decay together, and test no learning-rate schedule.
They do not establish a neural performance ceiling.
The initial GRU's inspected neighbours show some understandable grouping: the Oslo-area query matches other densely mapped cells, including a Trondheim-area cell.
The eligible query north of Tromsø and its closest matches have few mapped buildings and little recent activity.
This is exploratory interpretation from input months, not evidence of better prediction. The selected eligible cells do not necessarily cover the named city centres.
Each run contains `embedding-checks.json`, fitted PCA/probe tensors, and `trajectories.png`/`trajectories.svg` for validation cutoffs from December 2022 through June 2024.

The final neural model choice remains open. Reserved-test targets have not been inspected during baseline or neural development.

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

There is no remaining Milestone 3 blocker. [Milestone 4](SPEC.md#milestone-4--first-temporal-learner) is approved and in progress.
The baseline evidence supports this experiment, and Gate 3 fixes its forecast success criterion before neural training.

There is no tooling or GPU blocker. Both complete GRU runs are numerically stable, but neither provides the required forecast or representation improvement.
The [baseline-and-diagnosis campaign](#baseline-and-diagnosis-campaign) is complete. Linear activity summaries nearly match tree magnitude errors and improve occurrence ranking.
The paired residual-MLP capacity study above is complete: four scheduled runs, full validation exports, and the selected embedding comparison.
No configuration meets Gate 3, so no additional seeds were required. More capacity and longer training did not improve these configurations.
The next scope decision is the recommended bounded recency-weighting comparison; it has not started.
No new objective or recency-weighted refit is authorized. Spatial-context implementation remains deferred.
Milestone 4 still requires a final model choice, two additional seeds if a configuration becomes promising, and final reserved-test evaluation after that choice is frozen.
Geometry omissions, the approximate study boundary, and coarse cell-level aggregation remain limitations of the fixed experiment.
Reserved-test targets remain unavailable for development decisions until the final model choice is frozen.
