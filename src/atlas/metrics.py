"""The frozen Gate 2 magnitude and occurrence scores."""

from dataclasses import dataclass, field

import numpy as np
import torch

from atlas.samples import TARGET_NAMES

FAMILIES = {
    "raw": [i for i, name in enumerate(TARGET_NAMES) if name.startswith("edit_")],
    "semantic": [
        i for i, name in enumerate(TARGET_NAMES) if name.startswith("semantic_")
    ],
    "net": [i for i, name in enumerate(TARGET_NAMES) if name.startswith("net_")],
}


def transform(values: torch.Tensor) -> torch.Tensor:
    return values.sign() * values.abs().log1p()


def score_column(
    target: torch.Tensor,
    prediction_log: torch.Tensor,
    available: torch.Tensor,
    *,
    count_target: bool,
) -> tuple[float, float, int, int]:
    """Score one target/horizon; predictions are already in signed-log units.

    Return transformed MSE, tied-score AP, observations and positive support.
    Count projection in transformed space is equivalent to decoding and clamping.
    """
    if (
        target.ndim != 1
        or target.shape != prediction_log.shape
        or target.shape != available.shape
        or available.dtype != torch.bool
    ):
        raise ValueError("Scoring needs aligned vectors and a boolean target mask.")
    if (
        not torch.isfinite(prediction_log).all()
        or not torch.isfinite(target[available]).all()
    ):
        raise ValueError("Non-finite predictions or observed targets.")
    truth = target[available].double()
    prediction = prediction_log[available].double()
    if count_target:
        prediction = prediction.clamp_min(0)
    observed = len(truth)
    if not observed:
        return float("nan"), float("nan"), 0, 0
    mse = float((transform(truth) - prediction).square().mean())
    positive = truth.abs() >= 1
    support = int(positive.sum())
    if not support:
        return mse, float("nan"), observed, 0
    scores, order = prediction.abs().sort(descending=True)
    hits = positive[order].cumsum(0)
    # Evaluate precision only at the END of each tied score group.
    ends = torch.cat((scores[:-1] != scores[1:], scores.new_ones(1, dtype=torch.bool)))
    ranks = torch.arange(1, observed + 1, device=truth.device)[ends]
    hits = hits[ends].double()
    added_hits = torch.diff(hits, prepend=hits.new_zeros(1))
    ap = float((added_hits * hits / ranks).sum() / support)
    return mse, ap, observed, support


def _mean(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    total = np.where(valid, values, 0).sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def _number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


@dataclass
class Scores:
    """Small target-by-horizon table, including unavailable metric support."""

    mse: np.ndarray = field(default_factory=lambda: np.full((6, 37), np.nan))
    ap: np.ndarray = field(default_factory=lambda: np.full((6, 37), np.nan))
    observed: np.ndarray = field(default_factory=lambda: np.zeros((6, 37), dtype=int))
    positive: np.ndarray = field(default_factory=lambda: np.zeros((6, 37), dtype=int))

    def record(
        self, horizon: int, column: int, result: tuple[float, float, int, int]
    ) -> None:
        self.mse[horizon, column], self.ap[horizon, column] = result[:2]
        self.observed[horizon, column], self.positive[horizon, column] = result[2:]

    def summary(self) -> dict[str, object]:
        """Average within families, then equally across families and horizons."""
        family_mse = np.stack(
            [_mean(self.mse[:, columns], axis=1) for columns in FAMILIES.values()],
            axis=1,
        )
        family_ap = np.stack(
            [_mean(self.ap[:, columns], axis=1) for columns in FAMILIES.values()],
            axis=1,
        )

        def pair(mse: float, ap: float) -> dict[str, float | None]:
            return {
                "signed_log_rmse": _number(np.sqrt(mse)),
                "average_precision": _number(ap),
            }

        return {
            **pair(
                float(_mean(_mean(family_mse, 1))), float(_mean(_mean(family_ap, 1)))
            ),
            "horizons": [
                pair(float(_mean(family_mse[h])), float(_mean(family_ap[h])))
                for h in range(6)
            ],
            "families": {
                name: pair(
                    float(_mean(family_mse[:, i])), float(_mean(family_ap[:, i]))
                )
                for i, name in enumerate(FAMILIES)
            },
            "targets": {
                name: [
                    {
                        **pair(self.mse[h, c], self.ap[h, c]),
                        "observed": int(self.observed[h, c]),
                        "positive": int(self.positive[h, c]),
                    }
                    for h in range(6)
                ]
                for c, name in enumerate(TARGET_NAMES)
            },
        }
