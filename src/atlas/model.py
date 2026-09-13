"""A compact temporal encoder and the frozen, masked regression loss."""

import torch
from torch import nn

from atlas.metrics import FAMILIES, transform
from atlas.samples import INPUT_NAMES, TARGET_NAMES


class TemporalGRU(nn.Module):
    """Encode one cell history; predict all horizons from its final hidden state."""

    def __init__(self, hidden_size: int = 64, layers: int = 1) -> None:
        super().__init__()
        self.encoder = nn.GRU(
            len(INPUT_NAMES), hidden_size, num_layers=layers, batch_first=True
        )
        self.head = nn.Linear(hidden_size, 6 * len(TARGET_NAMES))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, hidden = self.encoder(inputs)
        embedding = hidden[-1]
        return self.head(embedding).reshape(-1, 6, len(TARGET_NAMES)), embedding


def family_mean(values: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    """Average a horizon/target table with the frozen family/horizon weights."""
    means, masks = [], []
    for columns in FAMILIES.values():
        counts = available[:, columns].sum(dim=1)
        means.append(
            torch.where(available[:, columns], values[:, columns], 0).sum(dim=1)
            / counts.clamp_min(1)
        )
        masks.append(counts > 0)
    valid = torch.stack(masks, dim=1)
    per_horizon = torch.stack(means, dim=1).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    horizons = valid.any(dim=1)
    return per_horizon.sum() / horizons.sum().clamp_min(1)


def masked_loss(
    prediction: torch.Tensor, target: torch.Tensor, available: torch.Tensor
) -> torch.Tensor:
    """Squared signed-log error; missing observations contribute no gradient.

    The linear head learns unconstrained transformed means. Count projection is
    applied during evaluation and decoding, as for the tree regressors.
    """
    truth = transform(torch.where(available, target, 0))
    errors = torch.where(available, prediction - truth, 0).square()
    counts = available.sum(dim=0)
    return family_mean(errors.sum(dim=0) / counts.clamp_min(1), counts > 0)


def project_counts(prediction: torch.Tensor) -> torch.Tensor:
    result = prediction.clone()
    columns = FAMILIES["raw"] + FAMILIES["semantic"]
    result[..., columns] = result[..., columns].clamp_min(0)
    return result
