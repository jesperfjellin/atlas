"""Bounded LightGBM input loading and numeric regression trees on PyTorch."""

import numpy as np
import torch

# Loading LightGBM first selects the compiler SDK's older HIP runtime.
import lightgbm as lgb  # isort: skip

from atlas.samples import WindowDataset


class InputSequence(lgb.Sequence):
    """Let LightGBM bin flattened histories without a full dense input matrix."""

    batch_size = 2048

    def __init__(self, dataset: WindowDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int | np.integer | slice | list[int]) -> np.ndarray:
        index = idx
        if isinstance(index, (int, np.integer)):
            indices = np.array([index])
        elif isinstance(index, slice):
            indices = np.arange(*index.indices(len(self)))
        else:
            indices = np.asarray(index)
        values = self.dataset.input_batch(indices).reshape(len(indices), -1)
        # LightGBM's sampled-row Sequence interface requires float64.
        return (
            values[0].astype(np.float64)
            if isinstance(index, (int, np.integer))
            else values.astype(np.float64)
        )


class RegressionTrees(torch.nn.Module):
    """Evaluate this experiment's LightGBM trees using tensor operations.

    LightGBM's public predict API uses host execution. This small numeric-tree
    evaluator keeps production forecasts on the AMD GPU as required by SPEC.
    """

    feature: torch.Tensor
    threshold: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    leaves: torch.Tensor
    root: torch.Tensor

    def __init__(self, model_text: str) -> None:
        super().__init__()
        trees: list[dict[str, str]] = []
        for line in model_text.splitlines():
            if line.startswith("Tree="):
                trees.append({})
            elif trees and "=" in line:
                key, value = line.split("=", 1)
                trees[-1][key] = value
        if not trees:
            raise ValueError("No regression trees in checkpoint.")
        max_nodes = max(int(tree["num_leaves"]) - 1 for tree in trees)
        max_leaves = max(int(tree["num_leaves"]) for tree in trees)
        shape = (len(trees), max(1, max_nodes))
        feature = np.zeros(shape, dtype=np.int64)
        threshold = np.zeros(shape, dtype=np.float64)
        left = np.full(shape, -1, dtype=np.int64)
        right = np.full(shape, -1, dtype=np.int64)
        leaves = np.zeros((len(trees), max_leaves), dtype=np.float64)
        self.depth = 0
        for i, tree in enumerate(trees):
            n = int(tree["num_leaves"]) - 1
            if int(tree.get("num_cat", "0")) or int(tree.get("is_linear", "0")):
                raise ValueError(
                    "Only numeric constant-leaf regression trees are supported."
                )
            for key, array in (
                ("split_feature", feature),
                ("threshold", threshold),
                ("left_child", left),
                ("right_child", right),
            ):
                array[i, :n] = np.fromstring(tree[key], sep=" ", dtype=array.dtype)
            leaves[i, : n + 1] = np.fromstring(tree["leaf_value"], sep=" ")

            def depth(node: int, tree_index: int = i) -> int:
                if node < 0:
                    return 0
                return 1 + max(
                    depth(int(left[tree_index, node]), tree_index),
                    depth(int(right[tree_index, node]), tree_index),
                )

            self.depth = max(self.depth, depth(0) if n else 0)
        self.register_buffer("feature", torch.from_numpy(feature))
        self.register_buffer("threshold", torch.from_numpy(threshold))
        self.register_buffer("left", torch.from_numpy(left))
        self.register_buffer("right", torch.from_numpy(right))
        self.register_buffer("leaves", torch.from_numpy(leaves))
        self.register_buffer(
            "root", torch.tensor([0 if int(t["num_leaves"]) > 1 else -1 for t in trees])
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return transformed regression predictions for finite flattened inputs."""
        row = torch.arange(len(inputs), device=inputs.device)[:, None]
        tree = torch.arange(len(self.root), device=inputs.device)[None, :]
        node = self.root.expand(len(inputs), -1)
        for _ in range(self.depth):
            safe = node.clamp_min(0)
            goes_left = (
                inputs[row, self.feature[tree, safe]] <= self.threshold[tree, safe]
            )
            child = torch.where(
                goes_left, self.left[tree, safe], self.right[tree, safe]
            )
            node = torch.where(node < 0, node, child)
        return self.leaves[tree, -node - 1].sum(dim=1)
