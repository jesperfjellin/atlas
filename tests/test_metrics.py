"""Protect the frozen scores against silent changes to the experiment result."""

import numpy as np
import pytest
import torch

from atlas.metrics import Scores, score_column, transform


def test_scores_group_ties_mask_missing_and_preserve_signed_net_changes() -> None:
    target = torch.tensor([2.0, 0.0, -3.0, 0.25, float("nan")])
    prediction = torch.tensor([1.0, 1.0, -0.5, 0.0, 99.0])
    mask = torch.tensor([True, True, True, True, False])
    mse, ap, count, positive = score_column(
        target, prediction, mask, count_target=False
    )
    # First tied group: one hit in two rows. Second group: two in three.
    assert ap == pytest.approx((0.5 + 2 / 3) / 2)
    assert (count, positive) == (4, 2)
    assert mse == pytest.approx(
        float(((transform(target[:4].double()) - prediction[:4]) ** 2).mean())
    )
    reverse = torch.tensor([3, 2, 1, 0, 4])
    assert score_column(
        target[reverse], prediction[reverse], mask[reverse], count_target=False
    ) == pytest.approx((mse, ap, count, positive))
    # Counts project negative predictions to zero; signed net targets do not.
    counted = score_column(target[:2], -prediction[:2], mask[:2], count_target=True)
    assert counted[0] == pytest.approx(float(np.log(3) ** 2 / 2))
    assert counted[1] == 0.5
    empty = score_column(target, prediction, torch.zeros_like(mask), count_target=False)
    assert np.isnan(empty[:2]).all() and empty[2:] == (0, 0)
    no_positive = score_column(
        target[3:4], prediction[3:4], mask[3:4], count_target=False
    )
    assert np.isnan(no_positive[1]) and no_positive[2:] == (1, 0)


def test_aggregation_weights_families_and_horizons_before_square_root() -> None:
    scores = Scores()
    scores.mse[:, :3] = 1
    scores.mse[:, 3:23] = 4
    scores.mse[:, 23:] = 9
    scores.mse[0, :3] = 16
    scores.ap[:, :3] = 0.2
    scores.ap[:, 3:23] = 0.4
    scores.ap[:, 23:] = 0.9
    # One target/horizon without positives must not become an AP of zero.
    scores.ap[2, 3] = np.nan
    result = scores.summary()
    assert result["signed_log_rmse"] == pytest.approx(np.sqrt((29 + 5 * 14) / 18))
    assert result["average_precision"] == pytest.approx(0.5)
    horizons = result["horizons"]
    assert isinstance(horizons, list)
    assert horizons[0]["signed_log_rmse"] == pytest.approx(np.sqrt(29 / 3))
    assert Scores().summary()["signed_log_rmse"] is None
    assert Scores().summary()["average_precision"] is None
