"""Numerical invariants of batched, masked full-data regression."""

import numpy as np
import pytest
import torch

from atlas.linear import Moments, ridge_solutions


@pytest.mark.parametrize("weighted", [False, True])
def test_batched_ridge_matches_augmented_least_squares_with_missing_targets(
    weighted: bool,
) -> None:
    rng = np.random.default_rng(37)
    x = rng.normal(size=(53, 7)) + np.arange(7) * 3
    x[:, -1] = 2  # Constant inputs and intercept must not make the solve singular.
    y = rng.normal(size=(53, 3)) + 5
    sample_weight = np.exp2(np.linspace(-4, 0, len(x))) if weighted else np.ones(len(x))
    mask = np.ones_like(y, dtype=bool)
    mask[::3, 1] = False
    projection = np.zeros((7, 5))
    projection[:5] = np.diag(1 / x[:, :5].std(0))
    for c in range(3):
        rows = np.flatnonzero(mask[:, c])
        moments = Moments.empty(7, 3, torch.tensor(0).device)
        for batch in np.array_split(rows, 4):
            moments.add(
                torch.from_numpy(x[batch]),
                torch.from_numpy(y[batch]),
                torch.from_numpy(sample_weight[batch]) if weighted else None,
            )
        for ridge, (w, b) in zip(
            (0.001, 0.1, 10.0),
            ridge_solutions(moments, torch.from_numpy(projection), [0.001, 0.1, 10.0]),
            strict=True,
        ):
            design = np.column_stack((x[rows] @ projection, np.ones(len(rows))))
            root_weight = np.sqrt(sample_weight[rows])
            penalty = np.diag([np.sqrt(ridge * sample_weight[rows].sum())] * 5 + [0.0])
            # Independent QR/SVD reference, with an unpenalized intercept.
            reference = np.linalg.lstsq(
                np.vstack((design * root_weight[:, None], penalty)),
                np.concatenate((y[rows, c] * root_weight, np.zeros(6))),
                rcond=None,
            )[0]
            expected = np.column_stack((x @ projection, np.ones(len(x)))) @ reference
            actual = x @ w[:, c].numpy() + b[c].item()
            np.testing.assert_allclose(actual, expected, atol=1e-10)
    repeated = Moments.empty(7, 3, torch.tensor(0).device)
    repeated.add(
        torch.from_numpy(np.tile(x, (2, 1))), torch.from_numpy(np.tile(y, (2, 1)))
    )
    full = Moments.empty(7, 3, torch.tensor(0).device)
    full.add(torch.from_numpy(x), torch.from_numpy(y))
    for a, b in zip(
        ridge_solutions(repeated, torch.from_numpy(projection), [0.1])[0],
        ridge_solutions(full, torch.from_numpy(projection), [0.1])[0],
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
