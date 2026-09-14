"""Protect probability objectives and comparisons from silent numerical errors."""

import math

import pytest
import torch

from atlas.occurrence import (
    logistic_fit,
    occurrence_loss,
    probability_scores,
    top_budget,
)


def test_occurrence_loss_preserves_labels_masks_and_family_weights() -> None:
    logits = torch.zeros((2, 6, 37), dtype=torch.float64, requires_grad=True)
    target = torch.zeros_like(logits)
    mask = torch.ones_like(logits, dtype=torch.bool)
    with torch.no_grad():
        logits[:, :, :3] = math.log(3)
        target[:, :, :3] = 2
        logits[:, :, 3:23] = -math.log(3)
        target[:, :, 23:] = -1  # Signed net decreases are positive events.
        target[0, 0, 0] = float("nan")
        logits[0, 0, 0] = float("nan")
        mask[0, 0, 0] = False
    loss = occurrence_loss(logits, target, mask)
    assert loss.item() == pytest.approx((-2 * math.log(0.75) + math.log(2)) / 3)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 0] == 0
    assert logits.grad[1, 0, 0].item() == pytest.approx(-0.25 / (6 * 3 * 3))
    assert logits.grad[0, 0, 3].item() == pytest.approx(0.25 / (2 * 6 * 3 * 20))


def test_probability_scores_and_inspection_ties_have_exact_expected_values() -> None:
    target = torch.tensor([1.0, 0.0, -2.0, 0.0, float("nan")], dtype=torch.float64)[
        :, None, None
    ].expand(5, 6, 37)
    p = torch.tensor([0.8, 0.8, 0.2, 0.1, 0.99], dtype=torch.float64)[
        :, None, None
    ].expand_as(target)
    mask = torch.isfinite(target)
    scores = probability_scores(target, mask, p)
    assert scores["average_precision"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert scores["brier"] == pytest.approx((0.04 + 0.64 + 0.64 + 0.01) / 4)
    assert scores["log_loss"] == pytest.approx(-math.log(0.8 * 0.2 * 0.2 * 0.9) / 4)
    top = top_budget(target[:4, 0, 0].abs() >= 1, p[:4, 0, 0], 1)
    assert top == {
        "inspected": 1,
        "expected_hits": 0.5,
        "precision": 0.5,
        "recall": 0.25,
    }
    # Fully tied forecasts have exactly random expected performance.
    assert (
        top_budget(torch.tensor([True, False, True, False]), torch.ones(4), 3)[
            "expected_hits"
        ]
        == 1.5
    )
    no_events = probability_scores(torch.zeros_like(target), mask, p)
    assert no_events["average_precision"] is None and no_events["brier"] is not None
    empty = probability_scores(target, torch.zeros_like(mask), p)
    assert empty["log_loss"] is None


def test_logistic_optimum_masks_rows_and_leaves_intercepts_unpenalized() -> None:
    x = torch.tensor([-1.0] * 4 + [1.0] * 4 + [99.0], dtype=torch.float64)[:, None]
    x = x.expand(-1, 2).clone()  # Exact collinearity exercises the rotation.
    y = torch.tensor(
        [
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [99, 99, 99],
        ],
        dtype=torch.float64,
    )
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[-1] = False
    fit = logistic_fit(x, y, mask, 0.1, tolerance=1e-7)
    w, b = fit["weights"], fit["intercept"]
    assert isinstance(w, torch.Tensor) and isinstance(b, torch.Tensor)
    # Analytic first-order condition for the symmetric regularized slope.
    torch.testing.assert_close(w[0], w[1])
    assert (w[:, 0].sum().sigmoid() - 0.75 + 0.1 * w[0, 0]).item() == pytest.approx(
        0, abs=5e-7
    )
    assert b[0].item() == pytest.approx(0, abs=5e-7)
    assert w[0, 2].item() == pytest.approx(0, abs=5e-7)
    assert b[2].sigmoid().item() == pytest.approx(0.25, abs=5e-7)
    assert w[0, 1] == 0
    assert b[1].sigmoid().item() == pytest.approx(0.5 / 9)
    assert fit["constant_outputs"] == 1
