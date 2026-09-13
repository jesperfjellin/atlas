"""Protect saved tree branch semantics and prediction sums."""

import torch

from atlas.trees import RegressionTrees


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
