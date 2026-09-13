"""Protect saved tree branch semantics and prediction sums."""

import numpy as np
import pytest
import torch

from atlas.trees import RegressionTrees, check_l2_updates


def test_l2_checkpoint_rejects_impossible_first_and_later_updates() -> None:
    labels = np.array([0.0, 2.0], dtype=np.float32)
    valid = "Tree=0\nleaf_value=0.5 1.5\nTree=1\nleaf_value=-0.75 0.75\n"
    check_l2_updates(valid, labels, 0.5)
    check_l2_updates("Tree=0\nleaf_value=-1 1\n", labels - 1, 1.0)
    # The real failure produced negative and enormous first-step count leaves.
    # Later trees must also respect the possible residuals of earlier trees.
    for corrupt in (
        "Tree=0\nleaf_value=-26.46 79.37\n",
        valid + "Tree=2\nleaf_value=25\n",
    ):
        with pytest.raises(ValueError):
            check_l2_updates(corrupt, labels, 0.5)


def test_saved_trees_preserve_thresholds_branches_and_constant_trees() -> None:
    forest = RegressionTrees("""tree
Tree=0
num_leaves=3
num_cat=0
split_feature=0 1
threshold=1.0000000001 2
left_child=-1 -2
right_child=1 -3
leaf_value=-2 3 7
is_linear=0
Tree=1
num_leaves=1
num_cat=0
split_feature=
threshold=
left_child=
right_child=
leaf_value=0.25
is_linear=0
""")
    # The first value is ABOVE the exact threshold. Rounding the threshold to
    # float32 first would silently select a different branch near this boundary.
    inputs = torch.tensor(
        [[1.0000000002, 2], [0, 4], [2, 3], [2, 1]], dtype=torch.float64
    )
    torch.testing.assert_close(
        forest(inputs), torch.tensor([3.25, -1.75, 7.25, 3.25], dtype=torch.float64)
    )
