# Changing the H3 cell resolution

Resolution 6 remains frozen for the current experiment under [Gate 2](../SPEC.md#gate-2-decisions--frozen-2026-09-13).
Use this guide when the owner approves a later experiment with smaller cells.
Increasing the H3 resolution makes cells smaller.

Keep resolution as a data-build parameter. Reuse the history reader, counting logic, sample loader, and model code.
A different grid requires fresh spatial aggregates, preprocessing, and training. The existing monthly totals cannot recover locations within a cell.
Targeted changes to spatial assignment or memory handling may be needed; do not introduce a new pipeline or configuration framework in advance.

## Code and configuration to inspect

These are the current entry points. Locate their equivalents if the code has moved.

| Concern | Current location |
| --- | --- |
| Norway grid and partitions | [`configs/split.yaml`](../configs/split.yaml), `read_split()` in [`splits.py`](../src/atlas/splits.py) |
| Corpus source, split path, output, and batch sizes | [`configs/norway.toml`](../configs/norway.toml), [`corpus.py`](../src/atlas/corpus.py) |
| Small development slice | [`configs/kristiansand.toml`](../configs/kristiansand.toml), [`dataset.py`](../src/atlas/dataset.py) |
| Entity-to-cell assignment and accumulation | `assign()` in [`features.py`](../src/atlas/features.py), [`accumulate.py`](../src/atlas/accumulate.py) |
| Input normalization and temporal samples | `CellMonths`, `Preprocessing`, and `WindowDataset` in [`samples.py`](../src/atlas/samples.py) |
| Baseline inputs, caches, and output paths | [`configs/baselines.toml`](../configs/baselines.toml), [`baselines.py`](../src/atlas/baselines.py) |
| Neural training, checkpoints, and embeddings | [`configs/gru.toml`](../configs/gru.toml), [`training.py`](../src/atlas/training.py), [`embeddings.py`](../src/atlas/embeddings.py) |

## Procedure

1. **Define the new experiment.** Record the approved resolution in `SPEC.md` and the experiment's status in `PROGRESS.md`.
   Preserve the existing data and model results.
   Keep the time windows, taxonomy, and feature definitions unless the owner also approves changing them.

2. **Create separate configuration and output paths.** For example, use `split-r7.yaml`, `norway-r7.toml`, and `baselines-r7.toml` in `configs/`.
   These names are examples; the files do not exist yet.
   Change `h3_resolution` in the new split file, then point the new corpus configuration's `split` field to it.
   The Norway corpus gets its resolution from that split; the small development-slice configuration has its own `h3_resolution` field.
   Choose new corpus and run directories. Keep configuration paths relative to their configuration file.

3. **Freeze the new grid and split before inspecting targets.** Generate the complete cell set from the geographic boundary at the new resolution.
   Retain empty cells; do not select cells from observed or future activity.
   Preserve the existing coarse geographic holdout parents and temporal dates where they remain valid.
   Recompute geographic groups and buffers. Review `buffer_rings`: a ring of smaller cells provides a narrower physical gap.
   Carry forward the explicitly designated development-area exclusions, currently the listed Kristiansand cells, at the new resolution.
   Include new cells that overlap those excluded areas.
   `Split.eligible_cells()` currently checks exact IDs: leaving resolution-6 IDs in `development_cells` does not exclude finer cells automatically.
   Verify the exclusions against the inspected geographic footprint.
   If reserved-test results have informed this change, they are now development evidence; agree on an untouched final test before claiming confirmation.

4. **Review spatial assignment and run a small pilot.** The [primary-cell rule](../SPEC.md#75-primary-cell-rule) assigns an entire entity's count, length, and area to one cell.
   Check whether this remains useful for roads and polygons that span many smaller cells.
   Changing resolution does not itself authorize intersection-weighted splitting; approve and document any change to feature meaning first.
   Use a fixed development area to check assignments, cross-cell transitions, row counts, and resource use before a full Norway build.
   Measure peak memory and runtime for corpus publication, loading, and a small GPU training run.
   `write_corpus()`, `CellMonths.read()`, and baseline `gpu_inputs()` currently allocate arrays across entire cell or validation sets.
   Neural scoring and embedding checks also retain predictions or representations across an evaluation set.
   More cells can require bounded loading at these points even though the model's per-sample feature dimensions stay the same.

5. **Rebuild from the original history.** Use the existing `.osh.pbf` file and the new corpus configuration through Compose:

   ```bash
   docker compose run --rm atlas uv run atlas build-dataset --config configs/norway-r7.toml
   ```

   Run this only after the new configuration and full-build scope are approved.
   Recompute entity assignments, opening state, transitions, and monthly features at the new resolution.
   Do not subdivide the old Parquet totals or copy old `parts/`, completion markers, or `build.json` into the new output.
   The build definition rejects a changed grid on resume. Keep that check intact.
   The current builder also writes reference arrays inside its output; let the new build create them through the existing path.

6. **Prepare fresh samples and normalization.** Load the new corpus with `CellMonths.read()` and the new split.
   Fit and save `Preprocessing` from training input cell-months only, then use it with `WindowDataset`.
   Generate development summaries from training and validation only. Keep reserved-test inspection within the structural checks permitted by [Section 8](../SPEC.md#8-development-and-reserved-test-data).
   Changing resolution alone does not change the feature count or temporal window lengths.
   Derive cell counts and sample counts from the new dataset; do not carry over the current Norway counts as constants.

7. **Retrain and establish a new benchmark.** Point the new baseline configuration's `dataset`, `split`, `preprocessing`, and `output` fields to the new paths.
   Use the existing Compose baseline command with that configuration, followed by `atlas train --config` with a new neural configuration.
   Do not reuse old normalization files, LightGBM bins, checkpoints, predictions, or embeddings as outputs of the new experiment.
   Run all three baselines again. Scores at different resolutions describe different targets and are not directly comparable as model improvements.
   Reapply the approved relative success criterion against the new baselines; do not carry over the old absolute score values.

8. **Verify and record the result.** Run the repository's required checks and inspect a small real-data example at the new resolution.
   Check the new cell coverage, temporal alignment, geographic exclusions, feature availability, and basic training resume.
   Add focused regression tests only where changed spatial or ML logic needs protection under `AGENTS.md`.
   Record the new corpus, benchmark, resource limits, and approved decisions in `PROGRESS.md` and `SPEC.md`.
   Keep raw data and generated artifacts outside Git; commit and push the related code, configuration, and documentation together.
