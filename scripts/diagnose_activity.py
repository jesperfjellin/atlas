"""Select 48 development cases and decompose existing forecast errors.

Run from the repository root through Compose:
    uv run python scripts/diagnose_activity.py

This is an outcome-stratified investigation, never a training/evaluation sample.
Outputs belong to this experiment only; no models or preprocessing are fitted.
"""

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TypedDict

import h3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from atlas.baselines import target_column, write_json
from atlas.diagnosis import read_predictions, scalar_score
from atlas.features import FEATURE_NAMES
from atlas.metrics import FAMILIES, Scores, score_column, transform
from atlas.samples import TARGET_NAMES, CellMonths, Preprocessing, WindowDataset
from atlas.splits import read_split
from atlas.training import gpu_device

CORPUS = Path("data/derived/norway-2015-2025")
DEFAULT_OUTPUT = Path("runs/history-diagnostics")
PERIODS = (
    (2015, 2016),
    (2017, 2018),
    (2019, 2020),
    (2021, 2021),
    (2022, 2022),
    (2023, 2024),
)
THEMES = ("building", "road", "poi", "landuse")
SEED = 20260914


class ErrorRow(TypedDict):
    model: str
    validation: str
    year: int
    parent: str
    family: str
    target: str
    horizon: int
    observations: int
    squared_error_sum: float
    aggregate_mse_contribution: float
    aggregate_mse_fraction: float


def development_corpus() -> CellMonths:
    split = read_split(Path("configs/split.yaml"))
    split = replace(
        split, months=tuple(m for m in split.months if m < split.temporal["test"][0])
    )
    return CellMonths.read(CORPUS, split)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_cases(corpus: CellMonths, output: Path) -> None:
    split = corpus.split
    positions = [i for i, c in enumerate(split.cells) if split.groups[c] == "train"]
    values = corpus.values[positions].astype(np.float64)
    valid = corpus.available[positions].all(2)
    cells = np.array(split.cells)[positions]
    parents = np.array([h3.cell_to_parent(c, 3) for c in cells])
    raw = values[
        :, :, [FEATURE_NAMES.index(f"edit_{k}") for k in ("create", "modify", "delete")]
    ].sum(2)
    changes = values[:, :, [FEATURE_NAMES.index(n) for n in TARGET_NAMES]]
    quiet = (changes == 0).all(2)
    # Opening mapped density is available before each focal event month.
    density = (
        values[:, :, FEATURE_NAMES.index("state_entity_count")]
        - values[:, :, FEATURE_NAMES.index("net_entity_count")]
    )
    density_bin = np.digitize(density, [1000, 10000, 100000])
    themes = {
        theme: np.log1p(
            np.abs(
                values[
                    :,
                    :,
                    [
                        i
                        for i, name in enumerate(FEATURE_NAMES)
                        if not name.startswith("state_") and theme in name
                    ],
                ]
            )
        ).sum(2)
        for theme in THEMES
    }
    rng = np.random.default_rng(SEED)
    used: set[tuple[int, int]] = set()
    parent_uses: Counter[str] = Counter()
    rows: list[dict] = []
    strata: list[dict] = []
    years = np.array([m.year for m in split.months])
    seasons = np.array([m.month for m in split.months])

    def choose(mask: np.ndarray, match: tuple[int, int] | None = None):
        candidates = np.argwhere(mask)
        candidates = np.array([pair for pair in candidates if tuple(pair) not in used])
        if not len(candidates):
            raise ValueError(
                "An inspection stratum is empty; revise its declared scope."
            )
        count = len(candidates)
        match_level = "balanced_parent"
        if match is not None:
            a, b = match
            same_parent = parents[candidates[:, 0]] == parents[a]
            same_density = density_bin[tuple(candidates.T)] == density_bin[a, b]
            same_season = seasons[candidates[:, 1]] == seasons[b]
            for label, compatible in (
                ("parent_density_month", same_parent & same_density & same_season),
                ("parent_density", same_parent & same_density),
                ("parent", same_parent),
                ("unmatched", np.ones(len(candidates), dtype=bool)),
            ):
                if compatible.any():
                    candidates = candidates[compatible]
                    match_level = label
                    break
        else:
            usage = np.array([parent_uses[p] for p in parents[candidates[:, 0]]])
            candidates = candidates[usage == usage.min()]
        i, j = map(int, candidates[rng.integers(len(candidates))])
        used.add((i, j))
        parent_uses[parents[i]] += 1
        return (i, j), count, len(candidates), match_level

    for p, (first, last) in enumerate(PERIODS):
        eligible = valid & ((years >= first) & (years <= last))[None, :]
        positive = raw[eligible & (raw > 0)]
        low, high, burst = np.quantile(positive, [0.25, 0.75, 0.99])
        ordinary = eligible & (raw > 0) & (raw >= low) & (raw <= high)
        for theme in THEMES:
            v = themes[theme]
            threshold = float(np.quantile(v[eligible & (v > 0)], 0.99))
            ordinary &= v < threshold
        requests = [("raw_burst", "raw", raw, burst)] * 2
        for t in range(2):
            theme = THEMES[(p * 2 + t) % len(THEMES)]
            v = themes[theme]
            requests.append(
                (
                    "semantic_net_burst",
                    theme,
                    v,
                    float(np.quantile(v[eligible & (v > 0)], 0.99)),
                )
            )
        for k, (stratum, theme, activity, threshold) in enumerate(requests):
            mask = eligible & (activity > 0) & (activity >= threshold)
            pair, count, pool, match_level = choose(mask)
            burst_id = f"case-{len(rows) + 1:02d}"
            control = "ordinary" if k % 2 == 0 else "quiet"
            control_mask = ordinary if control == "ordinary" else eligible & quiet
            control_choice = choose(control_mask, pair)
            strata.append(
                {
                    "period": f"{first}-{last}",
                    "theme": theme,
                    "threshold": threshold,
                    "eligible_cell_months": int(eligible.sum()),
                    "burst_candidates": int(mask.sum()),
                    "control": control,
                    "control_candidates": int(control_mask.sum()),
                }
            )
            for label, choice, matched_to in (
                (stratum, (pair, count, pool, match_level), ""),
                (control, control_choice, burst_id),
            ):
                (i, j), count, pool, match_level = choice
                rows.append(
                    {
                        "case": f"case-{len(rows) + 1:02d}",
                        "cell": cells[i],
                        "month": split.months[j].date().isoformat(),
                        "period": f"{first}-{last}",
                        "stratum": label,
                        "theme": theme,
                        "parent": parents[i],
                        "opening_entities": density[i, j],
                        "density_bin": int(density_bin[i, j]),
                        "raw_edits": raw[i, j],
                        "theme_activity": themes[theme][i, j]
                        if theme != "raw"
                        else raw[i, j],
                        "matched_to": matched_to,
                        "match_level": match_level,
                        "unused_candidates": count,
                        "selection_pool": pool,
                    }
                )
    if len(rows) != 48 or len({r["parent"] for r in rows}) < 6:
        raise ValueError("Inspection does not cover 48 cases and six parents.")
    write_csv(output / "sample_manifest.csv", rows)
    write_json(
        output / "selection.json",
        {
            "seed": SEED,
            "strata": strata,
            "description": "Two cases per stratum and period. "
            "Bursts: 99th percentile among positive cell-months. Ordinary: positive "
            "quartiles 25–75, excluding all theme bursts. Quiet: all 37 changes zero. "
            "Theme score sums log1p(abs(change)) across its semantic and net columns. "
            "Prefer least-used parents for bursts. Match controls by parent, "
            "opening-density bin and month, relaxing explicitly when needed. "
            "These are diagnostic cases, "
            "not a representative sample or a new evaluation population.",
        },
    )
    print(f"Selected {len(rows)} cases across {len(parent_uses)} parents.", flush=True)


def decompose_errors(corpus: CellMonths, output: Path) -> None:
    device = gpu_device()
    preprocessing = Preprocessing.read(CORPUS / "preprocessing.npz")
    rows: list[ErrorRow] = []
    all_scores = {}
    family_for = {c: f for f, columns in FAMILIES.items() for c in columns}
    for group in ("temporal", "geographic"):
        dataset = WindowDataset(
            corpus, preprocessing, "validation", geographic=group == "geographic"
        )
        starts = np.tile(dataset.starts, len(dataset.positions))
        sample_parents = np.repeat(
            [h3.cell_to_parent(corpus.split.cells[p], 3) for p in dataset.positions],
            len(dataset.starts),
        )
        labels = np.unique(sample_parents)
        parent_index = torch.as_tensor(
            np.searchsorted(labels, sample_parents), device=device
        )
        models = {
            "zero": Path(f"runs/diagnosis-norway/original/{group}-zero.parquet"),
            "trees": Path(f"runs/diagnosis-norway/original/{group}-trees.parquet"),
            "summary_ridge": Path(
                f"runs/diagnosis-norway/original/{group}-summary-1.parquet"
            ),
            "summary_mlp": Path(f"runs/mlp-summary-small-seed20260913/{group}.parquet"),
        }
        for model, path in models.items():
            predictions = read_predictions(path, dataset, device)
            scores = Scores()
            contributions = torch.zeros(
                (len(dataset), 6), device=device, dtype=torch.float64
            )
            model_rows: list[ErrorRow] = []
            # This corpus has complete target support. Fail instead of silently
            # misallocating the official weight when a new corpus has empty groups.
            for h in range(6):
                target_year = np.array(
                    [corpus.split.months[s + h].year for s in starts]
                )
                for c, target in enumerate(TARGET_NAMES):
                    raw, mask = (
                        torch.as_tensor(v, device=device)
                        for v in target_column(dataset, h, c)
                    )
                    result = score_column(
                        raw,
                        predictions[:, h, c],
                        mask,
                        count_target=not target.startswith("net_"),
                    )
                    scores.record(h, c, result)
                    if not mask.all():
                        raise ValueError(
                            "This decomposition requires the current complete support."
                        )
                    family = family_for[c]
                    weight = 1 / (6 * len(FAMILIES) * len(FAMILIES[family]))
                    pred = predictions[:, h, c]
                    if family != "net":
                        pred = pred.clamp_min(0)
                    squared = (transform(raw.double()) - pred).square()
                    contribution = squared * weight / len(dataset)
                    contributions[:, h] += contribution
                    for year in np.unique(target_year):
                        selected = torch.as_tensor(target_year == year, device=device)
                        sums = torch.zeros(
                            len(labels), device=device, dtype=torch.float64
                        )
                        counts = torch.zeros_like(sums)
                        sums.scatter_add_(0, parent_index[selected], squared[selected])
                        counts.scatter_add_(
                            0,
                            parent_index[selected],
                            torch.ones_like(squared[selected]),
                        )
                        for parent, total, count in zip(
                            labels, sums.tolist(), counts.tolist(), strict=True
                        ):
                            model_rows.append(
                                {
                                    "model": model,
                                    "validation": group,
                                    "year": int(year),
                                    "parent": parent,
                                    "family": family,
                                    "target": target,
                                    "horizon": h + 1,
                                    "observations": int(count),
                                    "squared_error_sum": total,
                                    "aggregate_mse_contribution": total
                                    * weight
                                    / len(dataset),
                                    "aggregate_mse_fraction": 0.0,
                                }
                            )
            summary = scores.summary()
            expected = scalar_score(summary, "signed_log_rmse") ** 2
            actual = float(contributions.sum())
            if not np.isclose(actual, expected, rtol=1e-10):
                raise ValueError(
                    "Error contributions do not recover the frozen aggregate MSE."
                )
            saved_path = (
                path.with_suffix(".json")
                if model != "summary_mlp"
                else path.parent / "metrics.json"
            )
            saved = json.loads(saved_path.read_text())
            saved = (
                saved["scores"]["overall"] if model != "summary_mlp" else saved[group]
            )
            for metric in ("signed_log_rmse", "average_precision"):
                if not np.isclose(
                    scalar_score(summary, metric), saved[metric], rtol=1e-7, atol=1e-10
                ):
                    raise ValueError(f"Saved {model}/{group}/{metric} does not match.")
            for row in model_rows:
                row["aggregate_mse_fraction"] = (
                    row["aggregate_mse_contribution"] / expected
                )
            rows.extend(model_rows)
            all_scores[f"{group}/{model}"] = summary
            # Combine repeated forecasts of the same cell-month, preserving their
            # contributions to the official window-weighted score. No deduplication
            # or independent-sample interpretation of the benchmark is implied.
            n = len(dataset.starts)
            values = contributions.cpu().numpy().reshape(len(dataset.positions), n, 6)
            cell_months = []
            for i, position in enumerate(dataset.positions):
                by_month: Counter[int] = Counter()
                for j, start in enumerate(dataset.starts):
                    for h in range(6):
                        by_month[start + h] += values[i, j, h]
                cell_months.extend(
                    {
                        "cell": corpus.split.cells[position],
                        "month": corpus.split.months[m].date().isoformat(),
                        "aggregate_mse_contribution": v,
                    }
                    for m, v in sorted(by_month.items())
                )
            pq.write_table(
                pa.Table.from_pylist(cell_months),
                output / f"{group}-{model}-cell-errors.parquet",
                compression="zstd",
            )
            print(
                f"Verified {group}/{model}: RMSE {summary['signed_log_rmse']:.6f}; "
                f"AP {summary['average_precision']:.6f}.",
                flush=True,
            )
    write_csv(output / "error_decomposition.csv", rows)
    write_json(output / "scores.json", all_scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path("runs").resolve()):
        raise ValueError("Write diagnostic outputs beneath runs/.")
    args.output.mkdir(parents=True, exist_ok=True)
    corpus = development_corpus()
    select_cases(corpus, args.output)
    if not args.selection_only:
        decompose_errors(corpus, args.output)


if __name__ == "__main__":
    main()
