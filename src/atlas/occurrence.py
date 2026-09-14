"""Direct event probabilities, regularized logistic fits, and probability scores."""

import math
from typing import TypedDict

import numpy as np
import torch
from torch.nn import functional as F

from atlas.metrics import FAMILIES, _mean, _number, average_precision
from atlas.model import family_mean
from atlas.samples import TARGET_NAMES


def occurrence_loss(
    logits: torch.Tensor, target: torch.Tensor, available: torch.Tensor
) -> torch.Tensor:
    """Binary log loss for abs(raw target) >= 1, with frozen family weights."""
    labels = (target.abs() >= 1).to(logits.dtype)
    safe = torch.where(available, logits, 0)
    errors = F.binary_cross_entropy_with_logits(safe, labels, reduction="none")
    counts = available.sum(0)
    totals = torch.where(available, errors, 0).sum(0)
    return family_mean(totals / counts.clamp_min(1), counts > 0)


def logistic_fit(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    available: torch.Tensor,
    strength: float,
    *,
    max_iterations: int = 1000,
    tolerance: float = 1e-5,
) -> dict[str, torch.Tensor | float | int]:
    """Fit independent logistic outputs using full-data float64 GPU L-BFGS.

    Each objective is mean observed log loss plus strength/2 * ||weights||².
    Intercepts are unpenalized. Constant-label outputs use a Jeffreys-smoothed
    training frequency; unavailable outputs are rejected. No validation data
    enters optimization. The caller supplies training-fitted standardized inputs.
    """
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError("Logistic regularization must be finite and positive.")
    if inputs.ndim != 2 or labels.ndim != 2 or labels.shape != available.shape:
        raise ValueError("Logistic inputs and observed binary labels must align.")
    if len(inputs) != len(labels) or not torch.isfinite(inputs).all():
        raise ValueError("Invalid logistic input rows.")
    counts = available.sum(0).double()
    if (counts == 0).any() or not (
        (labels[available] == 0) | (labels[available] == 1)
    ).all():
        raise ValueError("Every output needs observed binary training labels.")
    hits = torch.where(available, labels, 0).double().sum(0)
    active = (hits > 0) & (hits < counts)
    prior = (hits + 0.5) / (counts + 1)
    weights = torch.zeros(
        (inputs.shape[1], labels.shape[1]),
        dtype=torch.float64,
        device=inputs.device,
        requires_grad=True,
    )
    intercept = torch.zeros_like(prior, requires_grad=True)
    # Balance the optimization curvature of rare and common outputs. This is
    # a parameter change, not a change to the likelihood or regularization.
    variance = prior * (1 - prior)
    weight_scale = (variance + strength).rsqrt()
    intercept_scale = variance.rsqrt()
    initial_intercept = torch.logit(prior)
    optimizer = torch.optim.LBFGS(
        [weights, intercept],
        max_iter=max_iterations,
        tolerance_grad=tolerance,
        tolerance_change=1e-12,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    evaluations = 0

    def closure() -> torch.Tensor:
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        total = weights.new_zeros(())
        for offset in range(0, len(inputs), 32768):
            x = inputs[offset : offset + 32768].double()
            y = labels[offset : offset + 32768].double()
            mask = available[offset : offset + 32768] & active
            logits = (
                (x @ weights) * weight_scale
                + initial_intercept
                + intercept * intercept_scale
            )
            loss = (
                torch.where(
                    mask,
                    F.binary_cross_entropy_with_logits(
                        logits, torch.where(mask, y, 0), reduction="none"
                    ),
                    0,
                ).sum(0)
                / counts
            ).sum()
            loss.backward()
            total += loss.detach()
        penalty = strength / 2 * (weights * weight_scale).square().sum()
        penalty.backward()
        total += penalty.detach()
        if not torch.isfinite(total):
            raise ValueError("Non-finite logistic objective.")
        evaluations += 1
        return total

    optimizer.step(closure)
    final_loss = float(closure())
    assert weights.grad is not None and intercept.grad is not None
    gradient = max(
        float((weights.grad / weight_scale).abs().max()),
        float((intercept.grad / intercept_scale).abs().max()),
    )
    if gradient > tolerance * 5:
        raise ValueError(
            f"Logistic fit did not converge: maximum gradient {gradient:g}."
        )
    return {
        "weights": weights.detach() * weight_scale,
        "intercept": initial_intercept + intercept.detach() * intercept_scale,
        "strength": strength,
        "objective": final_loss,
        "max_gradient": gradient,
        "iterations": int(optimizer.state[weights]["n_iter"]),
        "evaluations": evaluations,
        "constant_outputs": int((~active).sum()),
    }


class Inspection(TypedDict):
    inspected: int
    expected_hits: float
    precision: float | None
    recall: float | None


def top_budget(positive: torch.Tensor, scores: torch.Tensor, budget: int) -> Inspection:
    """Expected hits at a fixed inspection budget, averaging boundary ties fairly."""
    count = min(budget, len(scores))
    if count < 1:
        return {"inspected": 0, "expected_hits": 0.0, "precision": None, "recall": None}
    threshold = scores.topk(count).values[-1]
    above, tied = scores > threshold, scores == threshold
    hits = float(positive[above].sum()) + (count - int(above.sum())) * float(
        positive[tied].float().mean()
    )
    support = int(positive.sum())
    return {
        "inspected": count,
        "expected_hits": hits,
        "precision": hits / count,
        "recall": hits / support if support else None,
    }


@torch.inference_mode()
def probability_scores(
    target: torch.Tensor,
    available: torch.Tensor,
    predictions: torch.Tensor,
    *,
    ranking_only: bool = False,
    calibration: bool = True,
) -> dict[str, object]:
    """Score aligned N×6×37 predictions; probability errors never become RMSE."""
    if (
        target.shape != predictions.shape
        or available.shape != target.shape
        or target.shape[1:] != (6, 37)
    ):
        raise ValueError("Probability scoring requires aligned N×6×37 arrays.")
    if (
        available.dtype != torch.bool
        or not torch.isfinite(predictions).all()
        or not torch.isfinite(target[available]).all()
    ):
        raise ValueError("Invalid predictions, observations, or probability masks.")
    if not ranking_only and ((predictions < 0) | (predictions > 1)).any():
        raise ValueError("Event probabilities must lie between zero and one.")
    tables = {
        name: np.full((6, 37), np.nan)
        for name in ("average_precision", "log_loss", "brier")
    }
    details: dict[str, list[dict[str, object]]] = {name: [] for name in TARGET_NAMES}
    for h in range(6):
        for c, name in enumerate(TARGET_NAMES):
            mask = available[:, h, c]
            p = predictions[mask, h, c].double()
            y = (target[mask, h, c].abs() >= 1).double()
            row: dict[str, object] = {"observed": len(p), "positive": int(y.sum())}
            if len(p):
                tables["average_precision"][h, c] = average_precision(y.bool(), p)
                if not ranking_only:
                    clipped = p.clamp(1e-7, 1 - 1e-7)
                    tables["log_loss"][h, c] = float(
                        -(y * clipped.log() + (1 - y) * (-clipped).log1p()).mean()
                    )
                    tables["brier"][h, c] = float((y - p).square().mean())
                    if calibration:
                        bins = (p * 10).long().clamp_max(9)
                        count = torch.bincount(bins, minlength=10)
                        sums = torch.bincount(bins, weights=p, minlength=10)
                        hits = torch.bincount(bins, weights=y, minlength=10)
                        row["calibration"] = [
                            {
                                "lower": b / 10,
                                "upper": (b + 1) / 10,
                                "observed": int(count[b]),
                                "mean_probability": float(sums[b] / count[b])
                                if count[b]
                                else None,
                                "event_fraction": float(hits[b] / count[b])
                                if count[b]
                                else None,
                            }
                            for b in range(10)
                        ]
            row.update(
                {
                    key: _number(values[h, c])
                    for key, values in tables.items()
                    if not ranking_only or key == "average_precision"
                }
            )
            details[name].append(row)
    overall, horizons, families = (
        {},
        [{} for _ in range(6)],
        {name: {} for name in FAMILIES},
    )
    for key, values in tables.items():
        if ranking_only and key != "average_precision":
            continue
        aggregated = np.stack(
            [_mean(values[:, cols], 1) for cols in FAMILIES.values()], 1
        )
        overall[key] = _number(float(_mean(_mean(aggregated, 1))))
        for h in range(6):
            horizons[h][key] = _number(float(_mean(aggregated[h])))
        for c, name in enumerate(FAMILIES):
            families[name][key] = _number(float(_mean(aggregated[:, c])))
    return {**overall, "horizons": horizons, "families": families, "targets": details}
