# Atlas — OSM Evolution Learner

**Status:** Lean proof-of-concept specification  
**Audience:** Project owner and coding agents  
**Version:** 0.3\
**Last updated:** 2026-09-12

## 1. How to use this specification

This specification defines the experiment we are trying to run and the few engineering rules needed to make its result trustworthy. It is not a request for a research platform, data platform, production service, or scientific reporting system.

Decision labels:

- **LOCKED** — implement as written unless the owner explicitly changes it.
- **PROVISIONAL** — use the stated starting point, then revisit it at the named gate.
- **DEFERRED** — do not implement or scaffold it yet.

The coding agent must work on only the current milestone. When a later decision is genuinely needed, ask for the smallest decision that unblocks the next piece of work.

## 2. Project thesis

> Build a small spatiotemporal machine-learning model that learns useful representations and patterns from how OpenStreetMap changes through time, using one consumer desktop.

The model studies **changes in OSM itself**. It does not need to decide whether an edit represents real construction, delayed mapping, an import, a correction, vandalism, or a tagging-convention change.

The machine-learning experiment is the project. OSM ingestion and feature preparation are necessary inputs to that experiment, not products in their own right.

## 3. What success means

The first useful result must answer two questions:

1. Can a small temporal neural model predict unseen future OSM changes better than sensible simple baselines?
2. Do its learned place-time embeddings contain useful structure beyond a simple low-dimensional representation of the same input features?

Evidence can be modest. This is a home proof of concept, not a scientific paper. A clear improvement on held-out data, interesting repeatable nearest neighbours or trajectories, and one simple representation comparison are enough to justify further work.

A negative result is also valid: the neural model may add nothing beyond recent-change rates or a boosted-tree model. If so, do not hide that result by adding a more elaborate pipeline.

## 4. The leanness rule

**LOCKED:** supporting infrastructure must remain substantially smaller and simpler than the OSM transformation and ML code it supports.

Before adding infrastructure, ask:

> Which currently approved experiment cannot be run without this code?

If there is no concrete answer, do not add it.

The initial project must not contain:

- an audit framework;
- a provenance or data-lineage framework;
- an experiment-tracking database;
- a data catalogue;
- a generic workflow engine;
- a job queue or message broker;
- a plugin system;
- a generic repository/service architecture;
- a formal ADR process;
- generated scientific reports;
- cloud infrastructure;
- a web API or frontend;
- code intended only for hypothetical planet-scale use.

Normal engineering records are enough:

- `uv.lock` for dependencies;
- one configuration file for a data build or training run;
- a small run directory containing the resolved configuration, metrics, logs, and checkpoint;
- a short console or Markdown summary at a decision gate.

Do not add checksums, Git-state capture, container digests, hardware archives, immutable run registries, artifact databases, or bit-for-bit reproduction machinery. We need to remember what was run and compare experiments; we do not need publication-grade reproducibility.

## 5. Scope and hard constraints

### 5.1 Compute

**LOCKED**

- Everything runs on one consumer desktop that is also used for ordinary work and gaming.
- No cluster, distributed training, or required cloud compute.
- No required stage may depend on NVIDIA CUDA.
- ML training and model execution use the AMD GPU through Docker Compose.
- ML commands must fail if GPU execution is unavailable. Do not implement CPU fallback.
- Models must fit comfortably in available memory.
- Training may run for days or weeks after short runs prove that the setup works.
- Long training jobs must checkpoint and resume without a general-purpose job-management system.

### 5.2 Geographic and data scope

**LOCKED**

- Norway is the initial study area.
- Predictive inputs and targets come from OSM history only.
- Geometry, tags, entity versions, timestamps, changesets, and aggregate editing activity are allowed.
- Calendar position and H3 relationships are allowed.
- Contributor usernames and individual contributor IDs are not model features.
- A distinct-contributor count may be used as an aggregate feature.
- OSM attribution and licence requirements must be included in the README.

**DEFERRED:** Scandinavia, selected global cities, Europe, and the planet.

### 5.3 Non-goals

The initial project does not attempt to:

- model the physical world independently of OSM;
- validate OSM against imagery, cadastral data, census data, or other sources;
- build a geospatial foundation model;
- use an LLM as the source of geographic learning;
- build a polished OSM analytics product;
- ingest live OSM replication changes;
- perform continual learning;
- use a graph neural network;
- provide a production application.

## 6. Initial learning experiment

### 6.1 Unit of learning

**LOCKED:** one H3 cell observed as a sequence of calendar months.

Months use UTC calendar intervals with `[start, end)` boundaries. Each interval includes its start and excludes its end.
An event at a month boundary belongs to the new month.

Use complete calendar months only. Do not use partial months for inputs or targets.

**PROVISIONAL defaults:**

- H3 resolution: 6;
- input window: 24 complete months;
- prediction window: the following 6 complete months;
- initial geography: Norway;
- initial model family: a compact GRU encoder with prediction heads.

These are ordinary configuration values, but do not build a generic configuration framework around them.

### 6.2 Fixed cell universe

Cell selection must not depend on whether a cell becomes active later in OSM history.

**LOCKED:** before constructing temporal samples, create one fixed study-cell set from the configured Norway boundary and H3 resolution. Include cells whose H3 centre lies inside the boundary. Use this same set for every period.

Consequences:

- A cell with no OSM entities yet is still a valid all-zero state.
- The model can observe and predict the first mapping activity in a previously empty cell.
- A historical sample is never included merely because the cell becomes active in the future.
- Sample eligibility may depend on source coverage through the required input and target periods, but not on whether an edit occurs in the target.

The boundary defines the fixed experiment domain; it is not a predictive feature. Do not build historical country-boundary reconstruction for this proof of concept.

### 6.3 Model input

For cell `c` and cutoff month `t`, the first model receives only:

- state features for months `t-23` through `t`;
- change features for months `t-23` through `t`;
- simple calendar encodings;
- a missingness mask where a value is genuinely unavailable.

The first model does not receive:

- any month after `t`;
- a learned cell-ID embedding;
- neighbouring-cell features;
- parent-cell features;
- contributor identities;
- external geographic data.

### 6.4 Prediction target

The target is the sequence of change features for months `t+1` through `t+6`. It must keep raw edit activity, semantic additions/removals, and net mapped-state change as separate target families.

Initial broad feature families should cover:

- buildings;
- roads, with a small number of broad road classes;
- POIs, with a small number of broad categories;
- land-use features;
- general OSM editing activity.

Start with a deliberately small taxonomy. Rare tags can map to `other`. Do not attempt to model the full OSM tagging universe.

## 7. Exact meaning of “change”

The data must keep three concepts separate. Do not collapse them into one ambiguous count.

### 7.1 Raw edit activity

A raw edit describes what happened to an OSM entity version:

- `edit_create`: no previous visible version, then a visible version;
- `edit_modify`: visible before and after, with a new entity version;
- `edit_delete`: visible before, then deleted or made non-visible.

Count one raw event per version transition.

Spatial attribution:

- create: assign to the entity's new primary cell;
- modify: assign once to the new primary cell;
- delete: assign to the previous primary cell.

If a modification moves an entity between cells, the raw modification is still counted only once, in the new cell.

### 7.2 Semantic additions and removals

For each broad semantic category, evaluate membership before and after an entity transition:

- not a member → member: `semantic_add`;
- member → not a member: `semantic_remove`;
- member of A → member of B: remove A and add B;
- member before and after in the same category: neither an addition nor a removal.

Therefore, removing `building=*` from an entity counts as a building semantic removal even if the entity itself remains in OSM. Adding it counts as a building semantic addition.

For a create, add every category the new entity belongs to. For a delete, remove every category the previous entity belonged to.

### 7.3 Movement between cells

If the primary cell changes during a modification:

- record a semantic removal from the old cell for every retained category;
- record a semantic addition to the new cell for every retained category;
- optionally expose simple `moved_out` and `moved_in` counts if useful;
- still count only one raw `edit_modify`, as described above.

This makes local mapped state reconcile even though the entity itself was not created or deleted globally.

### 7.4 Mapped state and net state change

At each month end, state features describe the reconstructable subset of visible OSM contents assigned to each cell.
Examples are building count, total mapped building area, and mapped road length.

Net state change is computed as:

```text
state_at_end_of_month - state_at_start_of_month
```

It is not inferred by summing raw edit counts.

A geometry modification within the same cell may therefore produce:

- one raw modification;
- no semantic addition or removal;
- a non-zero area or length change.

### 7.5 Primary-cell rule

For the first proof of concept:

- a node uses its coordinate;
- a line uses a deterministic midpoint along the geometry;
- a polygon uses a deterministic point on surface;
- a usable multipolygon/relation uses the equivalent point on its reconstructed geometry.

The entire entity count, area, or length is assigned to that primary cell. **Do not implement intersection-weighted splitting in the first version.** H3 resolution 6 is coarse enough for this simplification to be acceptable for the proof of concept.

If a snapshot or transition lacks reconstructable geometry, omit that snapshot or transition from cell features.

Computed feature values describe the reconstructable entity subset and remain numeric, including zero.
Skipping individual snapshots or transitions does not make an otherwise computed feature unavailable.

Use `null` with a false availability mask only for a cell-month feature that the builder cannot compute.
Missing source coverage is one example.

Record only two build counters:

- attempted geometry assignments;
- skipped geometry assignments due to unreconstructable geometry.

Do not build a quality-scoring subsystem around these counters.

The small taxonomy and these rules are frozen at Gate 1 after testing them on a real vertical slice.

## 8. Development and reserved test data

Leakage protection is one of the few scientific safeguards that must remain strict, because without it the ML result is meaningless.

### 8.1 Split timing

At the beginning of Milestone 2, after the usable source time range is known but **before inspecting target distributions**, create one small `configs/split.yaml` containing:

- train dates;
- validation dates;
- reserved temporal test dates;
- fixed geographic holdout groups;
- one random seed if selection is randomized.

Choose geographic holdouts from fixed cell IDs or coarser H3 parents, not from future OSM activity. Keep neighbouring cells in the same side of a geographic split.

All six target months of a sample must remain inside one temporal partition: training, validation, or reserved test.
Exclude samples whose target windows cross a temporal partition boundary.

Historical input months can precede the target partition because those observations were available at prediction time.
They must still precede the sample's first target month and obey the geographic holdout rules.

### 8.2 What may inspect the reserved test partition

The shared deterministic pipeline may build test rows and perform generic correctness checks such as:

- schema and dtype checks;
- expected date and cell coverage;
- absence of impossible timestamps;
- successful tensor loading.

Before final evaluation, do not inspect or summarize test:

- target distributions;
- semantic-category frequencies;
- sparsity by target;
- model metrics;
- prediction examples;
- embedding structure.

Choose taxonomy, feature pruning, target transforms, losses, hyperparameters, and success thresholds from train/validation data only.

This rule does not require an audit system. It requires a clear split file and disciplined code paths.

### 8.3 Other leakage rules

- Fit normalization and learned preprocessing on training data only.
- Do not select cells or samples based on target activity.
- Do not let a sample's input window include any of that sample's target months.
- Evaluation rows must preserve the natural zero-heavy distribution.
- Training may later resample or reweight examples if necessary, but validation and test data must remain unaltered.

## 9. Baselines and evaluation

### 9.1 Required baselines

Implement only these initially:

1. **Zero change:** predict no changes.
2. **Recent-rate persistence:** extrapolate a declared recent monthly rate.
3. **Boosted trees:** LightGBM or XGBoost using a flattened version of the same permitted inputs.

If a simple seasonal mean is obviously useful after looking at development data, it may be added. Do not build a large baseline framework.

### 9.2 Metrics

At Gate 2, select a small primary metric set using development data. It should normally contain:

- one magnitude metric for count or continuous targets;
- one occurrence metric for whether a broad change type happens;
- results by forecast month;
- an aggregate result and enough per-family detail to notice a completely failed target family.

Avoid dozens of metrics and elaborate dashboards.

### 9.3 Success-threshold timing

Gate 2 freezes:

- splits;
- feature and target definitions;
- target transforms and loss family;
- primary metrics.

Milestone 3 then implements and measures the baselines.

Gate 3 freezes a simple numeric definition of worthwhile improvement over the strongest baseline **before neural training begins**. It does not require bootstrap confidence intervals or a formal preregistration document.

If one neural run looks promising, repeat that configuration with two additional seeds before treating the improvement as real. Exploratory runs need only one seed.

### 9.4 Embedding evaluation

The first model must export one embedding per evaluated cell and cutoff month.

Initial embedding evaluation is deliberately small:

1. inspect nearest neighbours for selected place-time states;
2. visualize a few cell trajectories with PCA or UMAP;
3. run one simple linear probe and compare it with a dimension-matched PCA representation of the same inputs.

UMAP plots and hand-labelled interpretations are exploratory evidence, not proof by themselves. Do not build a general embedding-analysis suite.

## 10. Initial model

**PROVISIONAL:** a compact GRU encoder with one or more small prediction heads.

Starting range:

- 1–2 GRU layers;
- hidden/embedding dimension 64–128;
- comfortably below 5 million parameters.

Required capabilities:

- AMD GPU training through Docker Compose;
- configurable batch size;
- a clear embedding output;
- masked loss for genuinely unavailable values;
- checkpoint and resume;
- basic NaN/Inf detection;
- saved validation metrics.

Inspect development target distributions before choosing loss functions. Poisson or negative-binomial losses are candidates for counts; a transformed Huber loss is a candidate for continuous area or length targets. This is a Gate 2 decision, not a reason to build pluggable loss architecture.

## 11. Technology and repository shape

### 11.1 Naming

**LOCKED:** retain the existing internal Python package and CLI name `atlas`.

The eventual display or product name may change later. Do not rename the package during the proof of concept.

### 11.2 Stack

**LOCKED initial choices:**

- Python 3.14;
- `uv` for Python, dependencies, locking, and commands;
- PyTorch for the neural model;
- pytest, Ruff, `ty`, and pre-commit for development checks;
- H3 for spatial indexing;
- Parquet for cell-month datasets and exported embeddings;
- Docker and Docker Compose;
- `osmium`, pyosmium, or both where they directly simplify history extraction.

DuckDB may be used as an embedded library if it makes direct Parquet queries simpler. It must not become a service or data platform.

**DEFERRED:** PostgreSQL/PostGIS and Rust. Add either only after a measured blocker, not because the project might need it someday.

### 11.3 Storage

Use ordinary files:

```text
data/
  raw/          # downloaded or supplied OSM history
  derived/      # cell-month Parquet datasets
runs/
  <run-name>/   # config, metrics, checkpoint, embeddings, log
configs/
src/atlas/
tests/
compose.yaml
Dockerfile
pyproject.toml
uv.lock
```

There is no initial application database or experiment registry. A small JSON file beside a derived dataset may record only what is needed to interpret it: source filename, date range, H3 resolution, taxonomy version, and feature columns.

Raw and generated data remain outside Git. Avoid accidental overwrites, but do not build content-addressed or immutable storage machinery.

## 12. Runtime and CLI

### 12.1 Docker Compose

All supported workflows must run through Docker Compose. Start with one reusable application image and one service. Different commands may use the same service.

The main Compose service uses AMD ROCm on WSL and passes `/dev/dxg` into the container.
ML work always uses the AMD GPU. Do not add CPU/GPU profiles, overrides, or CPU fallback.
Validate GPU execution inside the container. The agent process does not need direct GPU device access.

### 12.2 Minimal CLI

The project should expose only the commands needed by the current milestones. The likely final shape is:

```text
atlas doctor
atlas build-dataset --config <path>
atlas train-baselines --config <path>
atlas train --config <path> [--resume <checkpoint>]
atlas evaluate --run <run-directory>
atlas explore-embeddings --run <run-directory>
```

Do not create empty commands for later milestones.

`atlas doctor` prints Python, PyTorch, GPU availability, and relevant tool versions, then checks a small GPU tensor operation and its gradient.
It must fail if ROCm or the AMD GPU is unavailable. It does not need to create a hardware inventory artifact.

Data downloading may initially be a documented manual step if that is simpler and more reliable than writing a downloader.

## 13. Long-running work

Implement resumability locally, where needed:

- data generation should write bounded chunks or partitions so a failed full run need not start from zero;
- training should keep `latest` and `best` checkpoints;
- training should save optimizer state, epoch/step, and configuration needed to resume;
- interruption should leave the last completed chunk or checkpoint usable.

Do not create a job manager, heartbeat service, scheduling database, or experiment queue.

## 14. Tests

Tests exist to protect the transformations and ML result, not to maximize coverage.

Create one tiny synthetic history fixture that covers:

- create, modify, and delete;
- adding and removing a semantic tag while the entity survives;
- a category transition;
- an entity moving between cells;
- a geometry-only modification;
- an event exactly on a month boundary;
- an incomplete month at a source boundary;
- a target window across a temporal partition boundary;
- an unavailable snapshot or transition geometry;
- missing source coverage for a cell-month feature;
- an empty cell that later receives its first OSM feature.

Essential automated checks:

- UTC month boundaries use `[start, end)`, and boundary events belong to the new month;
- samples contain no partial input or target months;
- each sample's complete target window stays inside one temporal partition;
- historical inputs can precede the target partition without violating the sample cutoff or geographic holdouts;
- raw edit counts, semantic transitions, and net state changes match fixture truth;
- cross-cell movement reconciles old and new state;
- skipped snapshots and transitions leave computed subset features numeric;
- an uncomputable cell-month feature uses `null` and a false availability mask;
- attempted and skipped geometry assignment counts match fixture truth;
- inputs contain no future months;
- fixed cell selection does not depend on later activity;
- normalization uses training data only;
- checkpoint resume works at a basic smoke-test level.

A manual acceptance check must verify that a small GPU training run completes and exports embeddings.

Do not build an exhaustive OSM conformance suite, property-testing framework, large integration-test harness, or test-only mini-platform. A small manual real-data check during Milestone 1 is sufficient.

## 15. Milestones and gates

### Milestone 0 — Minimal scaffold

Deliver:

- existing `atlas` package wired through `pyproject.toml`;
- `uv.lock`;
- Ruff, `ty`, pytest, and pre-commit configuration;
- one Dockerfile and one Compose service;
- a small `atlas doctor` command;
- minimal README commands;
- one smoke test.

Do not add a database, run registry, manifest layer, web app, data downloader, or empty future modules.

**Gate 0:** verify `atlas doctor` on the owner's AMD GPU through the main Compose service.

### Milestone 1 — Real historical-data vertical slice

Use a small Norwegian area and limited time range.

Mark this area and period as development data before inspecting it; it must not later become part of the reserved test partition.

Deliver:

- the smallest working path from one historical OSM source to cell-month features;
- before/after entity handling;
- the three change families from Section 7;
- the synthetic fixture tests;
- a concise printed or Markdown summary of attempted and skipped geometry assignments, runtime, memory, and a few manually inspected examples.

No formal audit report, source-comparison framework, data catalogue, or generic source-adapter system.

**Gate 1:** choose the history source/tool, usable time range, small taxonomy, primary-cell implementation, and final counting rules. Estimate whether a Norway run is practical.

### Milestone 2 — Norway development corpus

At the start of this milestone, create `configs/split.yaml` before inspecting target distributions.

Deliver:

- Norway H3 cell-month Parquet data;
- train/validation/reserved-test split;
- development-only summaries sufficient to choose features, transforms, losses, and metrics;
- an iterable PyTorch dataset or data loader;
- leakage tests.

Do not produce a report site, dataset registry, quality database, or publication-style analysis.

**Gate 2:** freeze the cell resolution, time windows, taxonomy/features, target definitions, split, transforms/loss family, and primary metrics. Do **not** freeze the numeric success threshold yet.

### Milestone 3 — Baselines

Deliver the zero-change, recent-rate, and boosted-tree baselines using the frozen development split and metrics.

Keep the evaluation output to a compact table plus saved metrics.

**Gate 3:** identify the strongest baseline and freeze the minimum neural-model improvement that would be worthwhile. Confirm that the problem has enough predictable signal to continue.

### Milestone 4 — First temporal learner

Deliver:

- compact GRU training and resume;
- validation predictions and metrics;
- exported embeddings;
- the three small embedding checks in Section 9.4;
- two additional seeds for a promising configuration;
- one final reserved-test evaluation only after the model choice is frozen.

**Gate 4:** stop, iterate within the same small model family, or approve one spatial-context experiment.

### Milestone 5 and later

**DEFERRED until Gate 4.** Consider only one addition at a time:

1. simple neighbouring-cell aggregates;
2. H3 parent-cell context;
3. learned neighbour pooling;
4. a small graph model;
5. offline continual-learning simulation;
6. live updates;
7. a read-only API and TypeScript inspection UI.

Each later addition needs a short new scope decision. Do not prebuild any of it.

## 16. Explicitly deferred decisions

Do not resolve these during Milestone 0:

- historical OSM source and exact usable years;
- final taxonomy;
- final H3 resolution;
- train/validation/test dates and geographic groups;
- loss functions and target transforms;
- numeric success threshold;
- neighbouring or multi-scale inputs;
- GNN architecture;
- continual-learning algorithm;
- live update cadence;
- PostgreSQL/PostGIS;
- Rust;
- API and frontend;
- deployment.

When one becomes necessary, present the current evidence, two or three realistic options, and a recommendation. A few paragraphs are enough.

## 17. Stop conditions

Pause and ask the owner if:

- a locked decision is technically infeasible;
- the source cannot reconstruct the required before/after state;
- Norway-wide storage or runtime appears unreasonable for the workstation;
- a shortcut would introduce target leakage;
- target-distribution information from the reserved test split has been exposed during development;
- a full Norway run is about to start before Gate 1;
- a deferred subsystem appears necessary;
- a destructive command would remove meaningful data or checkpoints.

Do not pause merely to propose extra architecture, reporting, abstraction, or audit work.

## 18. Immediate next instruction

> Implement Milestone 2 through Gate 2. Gate 1 and the full Norway build are approved. Use complete months from 2015 through 2025, with training targets in 2017–2022, validation in 2023–2024, and reserved temporal testing in 2025. Preserve the Kristiansand development exclusion. Complete the Norway corpus, PyTorch loader, leakage checks, and GPU batch acceptance before freezing Gate 2 decisions. Do not implement baselines, neural training, or supporting frameworks.
