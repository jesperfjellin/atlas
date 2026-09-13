"""Protect feature/target alignment, masks, and training-only preprocessing."""

from datetime import UTC, datetime

import numpy as np
import pytest
import torch

from atlas.features import FEATURE_NAMES, calendar_months
from atlas.samples import TARGET_COLUMNS, CellMonths, Preprocessing, WindowDataset
from atlas.splits import Split


@pytest.fixture
def corpus() -> CellMonths:
    def date(year: int) -> datetime:
        return datetime(year, 1, 1, tzinfo=UTC)

    cells = ("a_train", "b_empty", "c_validation", "d_test")
    months = tuple(calendar_months(date(2015), date(2026)))
    split = Split(
        cells,
        months,
        dict(zip(cells, ("train", "train", "validation", "test"), strict=True)),
        frozenset(),
        {
            "train": (date(2017), date(2023)),
            "validation": (date(2023), date(2025)),
            "test": (date(2025), date(2026)),
        },
        24,
        6,
        6,
    )
    values = np.zeros((4, len(months), len(FEATURE_NAMES)), dtype=np.float32)
    values[0] = (
        np.arange(len(months))[:, None] * 100 + np.arange(len(FEATURE_NAMES))[None, :]
    )
    values[2:] = 1e6
    available = np.ones_like(values, dtype=bool)
    available[0, 0, 0] = False
    available[0, 24, 2] = False
    values[~available] = 0
    return CellMonths(split, values, available)


def test_windows_align_columns_calendar_and_masks_without_future_inputs(
    corpus: CellMonths,
) -> None:
    transform = Preprocessing(
        np.zeros(51, dtype=np.float32), np.ones(51, dtype=np.float32)
    )
    dataset = WindowDataset(corpus, transform, "train")
    first = dataset[0]
    inputs, targets, mask = first["inputs"], first["targets"], first["target_mask"]
    assert (
        isinstance(inputs, torch.Tensor)
        and isinstance(targets, torch.Tensor)
        and isinstance(mask, torch.Tensor)
    )
    assert inputs.shape == (24, 104) and targets.shape == mask.shape == (6, 37)
    assert first["cutoff"] == "2016-12-01"
    np.testing.assert_array_equal(
        targets.numpy(), corpus.values[0, 24:30][:, TARGET_COLUMNS]
    )
    assert inputs[0, 0] == inputs[0, 53] == 0
    assert inputs[0, 51] == 0 and inputs[0, 52] == 1
    assert not mask[0, 2] and targets[0, 2] == 0
    original = inputs.clone()
    corpus.values[0, 24:] = 5e7
    assert torch.equal(dataset[0]["inputs"], original)
    empty = dataset[len(dataset.starts)]
    assert empty["cell"] == "b_empty"
    empty_inputs = empty["inputs"]
    assert isinstance(empty_inputs, torch.Tensor)
    assert torch.count_nonzero(empty_inputs[:, :51]) == 0
    assert empty_inputs[:, 53:].all()
    assert len(dataset) == 2 * 67


def test_preprocessing_ignores_heldout_cells_and_months(corpus: CellMonths) -> None:
    before = Preprocessing.fit(corpus)
    assert before.mean.any() and (before.scale > 0).all()
    corpus.values[2:] = -5e8
    corpus.available[2:] = False
    # Training targets can end in December 2022, but the final training input
    # cutoff is June 2022. Even later training targets must not fit input scaling.
    corpus.values[:2, 90:] = 9e9
    corpus.available[:2, 90:] = False
    after = Preprocessing.fit(corpus)
    np.testing.assert_array_equal(after.mean, before.mean)
    np.testing.assert_array_equal(after.scale, before.scale)


def test_baseline_batches_rates_and_labels_obey_the_same_cutoff(
    corpus: CellMonths,
) -> None:
    from atlas.baselines import recent_rate, target_column
    from atlas.metrics import transform
    from atlas.trees import InputSequence

    preprocessing = Preprocessing(
        np.zeros(51, dtype=np.float32), np.ones(51, dtype=np.float32)
    )
    dataset = WindowDataset(corpus, preprocessing, "train")
    sequence = InputSequence(dataset)
    # LightGBM's native Sequence sampler supplies NumPy integer scalars.
    np.testing.assert_array_equal(sequence[np.int64(0)], sequence[0])
    indices = np.array([67, 66, 0])
    batch = sequence[indices.tolist()].reshape(3, 24, 104)
    assert not batch[0, :, :51].any()
    np.testing.assert_allclose(batch[1, :, 0], np.log1p(corpus.values[0, 66:90, 0]))
    np.testing.assert_array_equal(batch[2], dataset[0]["inputs"].numpy())
    corpus.available[0, 18:24, 1] = False
    corpus.available[0, 22:24, 2] = False
    device = dataset[0]["inputs"].device
    rates = recent_rate(dataset, 6, device)
    torch.testing.assert_close(
        rates[0, 0], transform(torch.tensor(corpus.values[0, 18:24, 0].mean()))
    )
    assert rates[0, 1] == 0
    torch.testing.assert_close(
        rates[0, 2],
        transform(torch.tensor(corpus.values[0, 18:22, 2].mean())),
    )
    assert not rates[67:].any()
    values, mask = target_column(dataset, 5, 0)
    assert values[0] == corpus.values[0, 29, 0] and mask.all()
    corpus.values[0, 24:] = 1e8
    corpus.values[2:] = -1e8
    torch.testing.assert_close(recent_rate(dataset, 6, device)[0], rates[0])
