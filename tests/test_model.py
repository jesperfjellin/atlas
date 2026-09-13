"""Protect the neural loss and continuation of a fitted model."""

from pathlib import Path

import pytest
import torch

from atlas.model import TemporalGRU, masked_loss, project_counts
from atlas.training import restore_checkpoint, save_checkpoint


def test_loss_weights_targets_families_and_horizons_and_masks_gradients() -> None:
    prediction = torch.ones((2, 6, 37), requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(target, dtype=torch.bool)
    with torch.no_grad():
        prediction[:, :, 3:23] = 2
        prediction[:, :, 23:] = -3
        prediction[:, 0, :3] = 4
        target[0, 0, 0] = float("nan")
        mask[0, 0, 0] = False
    loss = masked_loss(prediction, target, mask)
    assert float(loss.detach()) == pytest.approx((29 + 5 * 14) / 18)
    loss.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert prediction.grad[0, 0, 0] == 0
    assert prediction.grad[1, 0, 0] == pytest.approx(8 / (6 * 3 * 3))
    assert prediction.grad[0, 1, 3] == pytest.approx(4 / (2 * 6 * 3 * 20))
    projected = project_counts(-prediction.detach())
    assert (projected[:, :, :23] == 0).all() and (projected[:, :, 23:] == 3).all()
    # Missing whole families/horizons must not become zero-loss observations.
    mask[:, :, 3:] = False
    mask[:, 0, :] = False
    assert masked_loss(prediction, target, mask).item() == pytest.approx(1)


def test_checkpoint_restores_embedding_optimizer_and_next_update(
    tmp_path: Path,
) -> None:
    torch.manual_seed(83)
    model = TemporalGRU()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    inputs = torch.randn(3, 24, 104)
    target = torch.zeros(3, 6, 37)
    mask = torch.ones_like(target, dtype=torch.bool)

    def update(learner: TemporalGRU, optim: torch.optim.Optimizer) -> None:
        optim.zero_grad(set_to_none=True)
        prediction, embedding = learner(inputs)
        assert prediction.shape == (3, 6, 37) and embedding.shape == (3, 64)
        masked_loss(prediction, target, mask).backward()
        optim.step()

    update(model, optimizer)
    before = model(inputs)
    path = tmp_path / "latest.pt"
    configuration: dict[str, object] = {"seed": 83}
    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=1,
        step=1,
        best_rmse=0.7,
        bad_epochs=0,
        configuration=configuration,
    )
    expected_random = torch.rand(3)
    update(model, optimizer)
    resumed = TemporalGRU()
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    assert restore_checkpoint(path, resumed, resumed_optimizer, configuration) == (
        1,
        1,
        0.7,
        0,
    )
    torch.testing.assert_close(torch.rand(3), expected_random)
    for actual, expected in zip(resumed(inputs), before, strict=True):
        torch.testing.assert_close(actual, expected)
    update(resumed, resumed_optimizer)
    for actual, expected in zip(resumed.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError):
        restore_checkpoint(path, resumed, resumed_optimizer, {"seed": 84})
