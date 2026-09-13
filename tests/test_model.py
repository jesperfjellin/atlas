"""Protect the neural loss and continuation of a fitted model."""

from pathlib import Path

import pytest
import torch

from atlas.model import ResidualMLP, TemporalGRU, masked_loss, project_counts
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


@pytest.mark.parametrize("kind", ["gru", "mlp"])
def test_checkpoint_restores_embedding_optimizer_and_next_update(
    tmp_path: Path,
    kind: str,
) -> None:
    torch.manual_seed(83)
    model = TemporalGRU() if kind == "gru" else ResidualMLP("summary", 16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    inputs = torch.randn(3, 24, 104) if kind == "gru" else torch.randn(3, 252)
    if isinstance(model, ResidualMLP):
        inputs[:, model.observed_columns] = 1
        model.fit_input_scaling(inputs)
    target = torch.zeros(3, 6, 37)
    mask = torch.ones_like(target, dtype=torch.bool)

    def update(
        learner: TemporalGRU | ResidualMLP, optim: torch.optim.Optimizer
    ) -> None:
        learner.train()
        optim.zero_grad(set_to_none=True)
        prediction, embedding = learner(inputs)
        assert prediction.shape == (3, 6, 37) and embedding.shape == (3, 64)
        masked_loss(prediction, target, mask).backward()
        optim.step()

    update(model, optimizer)
    before = model.eval()(inputs)
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
    resumed = TemporalGRU() if kind == "gru" else ResidualMLP("summary", 16)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    assert restore_checkpoint(path, resumed, resumed_optimizer, configuration) == (
        1,
        1,
        0.7,
        0,
    )
    torch.testing.assert_close(torch.rand(3), expected_random)
    for actual, expected in zip(resumed.eval()(inputs), before, strict=True):
        torch.testing.assert_close(actual, expected)
    update(resumed, resumed_optimizer)
    for actual, expected in zip(resumed.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError):
        restore_checkpoint(path, resumed, resumed_optimizer, {"seed": 84})


def test_mlp_scales_only_observed_summary_means_and_preserves_other_inputs() -> None:
    model = ResidualMLP("summary", 16)
    training = torch.zeros(3, 252)
    mean_column = int(model.mean_columns[0])
    observed_column = int(model.observed_columns[0])
    training[:, mean_column] = torch.tensor([2.0, 4.0, 1e8])
    training[:, observed_column] = torch.tensor([1.0, 1.0, 0.0])
    model.fit_input_scaling(training)
    assert model.summary_mean[0] == 3 and model.summary_scale[0] == 1
    validation = training.clone()
    validation[:, 0] = 123  # Frozen state normalization must not be refitted here.
    validation[:, mean_column] = torch.tensor([5.0, -3.0, -1e8])
    normalized = model.normalized_inputs(validation)
    torch.testing.assert_close(
        normalized[:, mean_column], torch.tensor([2.0, -6.0, 0.0])
    )
    assert torch.equal(normalized[:, 0], validation[:, 0])
    assert torch.equal(
        normalized[:, model.observed_columns], validation[:, model.observed_columns]
    )
    assert model.summary_mean[0] == 3 and model.summary_scale[0] == 1


def test_linear_probe_masks_each_target_and_fits_scaling_on_training_rows() -> None:
    from atlas.embeddings import fit_probe

    x = torch.linspace(-2, 2, 21, dtype=torch.float64)
    representation = torch.stack((x, torch.ones_like(x)), dim=1)
    expected = torch.stack((3 * x + 0.5, 1 - x, x), dim=1)
    raw = expected.sign() * expected.abs().expm1()
    mask = torch.ones_like(raw, dtype=torch.bool)
    mask[:4, 0] = False
    mask[:, 2] = False
    raw[~mask] = float("nan")
    probe = fit_probe(representation, raw, mask, ridge=1e-8)
    assert probe.mean[0].item() == pytest.approx(0, abs=1e-12)
    assert probe.scale[1] == 1
    prediction = probe.predict(torch.tensor([[3.0, 1.0]], dtype=torch.float64))
    torch.testing.assert_close(
        prediction,
        torch.tensor([[9.5, -2.0, 0.0]], dtype=torch.float64),
        atol=1e-6,
        rtol=1e-6,
    )
