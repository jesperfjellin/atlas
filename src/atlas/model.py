"""A compact temporal encoder and the frozen, masked regression loss."""

import torch
from torch import nn

from atlas.linear import CONTROL_COLUMNS, JOINT_SIZE, ORDERED_SIZE
from atlas.metrics import FAMILIES, transform
from atlas.samples import INPUT_NAMES, TARGET_NAMES


class TemporalGRU(nn.Module):
    """Encode one cell history; predict all horizons from its final hidden state."""

    def __init__(self, hidden_size: int = 64, layers: int = 1) -> None:
        super().__init__()
        self.embedding_size = hidden_size
        self.encoder = nn.GRU(
            len(INPUT_NAMES), hidden_size, num_layers=layers, batch_first=True
        )
        self.head = nn.Linear(hidden_size, 6 * len(TARGET_NAMES))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, hidden = self.encoder(inputs)
        embedding = hidden[-1]
        return self.head(embedding).reshape(-1, 6, len(TARGET_NAMES)), embedding


class ResidualMLP(nn.Module):
    """Two residual blocks for the approved summary/history capacity comparison."""

    input_columns: torch.Tensor
    mean_columns: torch.Tensor
    observed_columns: torch.Tensor
    summary_mean: torch.Tensor
    summary_scale: torch.Tensor

    def __init__(self, inputs: str, width: int, dropout: float = 0.1) -> None:
        super().__init__()
        if inputs not in {"summary", "history"} or width < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid residual MLP inputs, width, or dropout.")
        columns = (
            CONTROL_COLUMNS["summary"]
            if inputs == "summary"
            else list(range(JOINT_SIZE))
        )
        self.input_size, self.embedding_size = len(columns), 64
        self.register_buffer("input_columns", torch.tensor(columns))
        means = [ORDERED_SIZE + offset + c for offset in (0, 111) for c in range(37)]
        self.register_buffer(
            "mean_columns", torch.tensor([columns.index(c) for c in means])
        )
        self.register_buffer(
            "observed_columns", torch.tensor([columns.index(c + 74) for c in means])
        )
        self.register_buffer("summary_mean", torch.zeros(len(means)))
        self.register_buffer("summary_scale", torch.ones(len(means)))
        self.projection = nn.Sequential(nn.Linear(self.input_size, width), nn.GELU())
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(width),
                    nn.Linear(width, width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(width, width),
                    nn.Dropout(dropout),
                )
                for _ in range(2)
            ]
        )
        self.embedding = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64))
        self.head = nn.Linear(64, 6 * len(TARGET_NAMES))

    @torch.no_grad()
    def fit_input_scaling(self, training_inputs: torch.Tensor) -> None:
        """Fit only engineered means; keep frozen channels and fractions intact."""
        total = self.summary_mean.double().new_zeros(74)
        squares, counts = torch.zeros_like(total), torch.zeros_like(total)
        for offset in range(0, len(training_inputs), 4096):
            batch = training_inputs[offset : offset + 4096]
            available = batch[:, self.observed_columns] > 0
            values = torch.where(available, batch[:, self.mean_columns].double(), 0)
            total += values.sum(0)
            squares += values.square().sum(0)
            counts += available.sum(0)
        mean = total / counts.clamp_min(1)
        scale = (squares / counts.clamp_min(1) - mean.square()).clamp_min(0).sqrt()
        self.summary_mean.copy_(mean)
        self.summary_scale.copy_(torch.where(scale < 1e-6, 1, scale))

    def normalized_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs.clone()
        values[:, self.mean_columns] = torch.where(
            inputs[:, self.observed_columns] > 0,
            (inputs[:, self.mean_columns] - self.summary_mean) / self.summary_scale,
            0,
        )
        return values

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.projection(self.normalized_inputs(inputs))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        embedding = self.embedding(hidden)
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
