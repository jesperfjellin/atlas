"""Three small development checks for learned place-time representations."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import h3
import matplotlib
import numpy as np
import torch
import yaml

from atlas.baselines import write_json
from atlas.metrics import transform
from atlas.model import ResidualMLP, TemporalGRU
from atlas.samples import INPUT_NAMES, CellMonths, Preprocessing, WindowDataset
from atlas.splits import read_split
from atlas.training import gpu_device, model_inputs, sample_keys, score_predictions

matplotlib.use("Agg")


@dataclass
class LinearProbe:
    mean: torch.Tensor
    scale: torch.Tensor
    weights: torch.Tensor

    def predict(self, representation: torch.Tensor) -> torch.Tensor:
        x = (representation.double() - self.mean) / self.scale
        return x @ self.weights[:-1] + self.weights[-1]


def fit_probe(
    representation: torch.Tensor,
    target: torch.Tensor,
    available: torch.Tensor,
    ridge: float = 0.01,
) -> LinearProbe:
    """Fit a standardized ridge probe on training rows, masking each target.

    The intercept is unpenalized. Ridge is relative to the observed sample count
    so the declared strength does not depend on the size of the training subset.
    """
    x = representation.double()
    mean, scale = x.mean(dim=0), x.std(dim=0, correction=0)
    scale = torch.where(scale < 1e-6, 1, scale)
    x = torch.cat(((x - mean) / scale, x.new_ones((len(x), 1))), dim=1)
    y = transform(torch.where(available, target, 0)).double()
    penalty = ridge * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    penalty[-1, -1] = 0
    weights = x.new_zeros((x.shape[1], y.shape[1]))
    if available.all():
        weights = torch.linalg.solve(x.T @ x / len(x) + penalty, x.T @ y / len(x))
    else:
        for c in range(y.shape[1]):
            selected = available[:, c]
            if selected.any():
                observed = x[selected]
                weights[:, c] = torch.linalg.solve(
                    observed.T @ observed / len(observed) + penalty,
                    observed.T @ y[selected, c] / len(observed),
                )
    return LinearProbe(mean, scale, weights)


@torch.inference_mode()
def representations(
    model: TemporalGRU | ResidualMLP,
    dataset: WindowDataset,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect a bounded, target-independent training subset for these checks."""
    dimensions = (
        model.input_size if isinstance(model, ResidualMLP) else 24 * len(INPUT_NAMES)
    )
    inputs = torch.empty((len(indices), dimensions), device=device)
    embeddings = torch.empty((len(indices), model.embedding_size), device=device)
    for offset in range(0, len(indices), 1024):
        selected = indices[offset : offset + 1024]
        values = model_inputs(model, dataset, selected, device)
        inputs[offset : offset + len(selected)] = (
            model.normalized_inputs(values)
            if isinstance(model, ResidualMLP)
            else values.flatten(1)
        )
        embeddings[offset : offset + len(selected)] = model(values)[1]
    return inputs, embeddings


def selected_cells(dataset: WindowDataset) -> list[tuple[str, int]]:
    """Pick study cells nearest three declared cities, without using activity."""
    coordinates = np.array(
        [h3.cell_to_latlng(dataset.corpus.split.cells[p]) for p in dataset.positions]
    )
    selected = []
    for label, latitude, longitude in (
        ("Kristiansand", 58.15, 8.0),
        ("Oslo", 59.91, 10.75),
        ("Tromsø", 69.65, 18.96),
    ):
        distances = (coordinates[:, 0] - latitude) ** 2 + (
            (coordinates[:, 1] - longitude) * np.cos(np.deg2rad(latitude))
        ) ** 2
        selected.append((label, int(distances.argmin())))
    return selected


@torch.inference_mode()
def explore_embeddings(run: Path) -> None:
    """Compare a saved neural encoder with PCA on development data only."""
    import matplotlib.pyplot as plt

    device = gpu_device()
    run = run.resolve()
    config = json.loads((run / "config.json").read_text())
    settings = config["training"]
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
    if checkpoint["configuration"] != config:
        raise ValueError("Checkpoint and run configuration differ.")
    split = read_split(Path(config["split"]))
    split = replace(
        split,
        months=tuple(m for m in split.months if m < split.temporal["validation"][1]),
    )
    corpus = CellMonths.read(Path(config["dataset"]), split)
    preprocessing = Preprocessing.read(Path(config["preprocessing"]))
    if (
        yaml.safe_load(Path(config["split"]).read_text())
        != config["split_configuration"]
        or preprocessing.mean.tolist() != config["input_mean"]
        or preprocessing.scale.tolist() != config["input_scale"]
    ):
        raise ValueError(
            "The split or input preprocessing differs from the trained run."
        )
    model = (
        (
            ResidualMLP(**config["model"])
            if "model" in config
            else TemporalGRU(settings["hidden_size"], settings["layers"])
        )
        .to(device)
        .eval()
    )
    encoder_name = "mlp" if isinstance(model, ResidualMLP) else "gru"
    model.load_state_dict(checkpoint["model"])
    training = WindowDataset(corpus, preprocessing, "train")
    # The same rows fit PCA, both probes, and representation scaling. No labels
    # choose these rows. Full, natural validation remains unchanged.
    seed, limit, ridge = 20260913, 32768, 0.01
    torch.manual_seed(seed)
    indices = np.random.default_rng(seed).choice(
        len(training), min(limit, len(training)), replace=False
    )
    inputs, encoded = representations(model, training, indices, device)
    raw_target, mask = training.target_batch(indices)
    target = torch.as_tensor(raw_target.reshape(len(indices), -1), device=device)
    available = torch.as_tensor(mask.reshape(len(indices), -1), device=device)
    dimension = model.embedding_size
    input_mean = inputs.mean(dim=0)
    _, _, components = torch.pca_lowrank(inputs, q=dimension + 16, center=True, niter=2)
    components = components[:, :dimension]
    pca = (inputs - input_mean) @ components
    probes = {
        encoder_name: fit_probe(encoded, target, available, ridge),
        "pca": fit_probe(pca, target, available, ridge),
    }
    embedding_mean = encoded.mean(dim=0)
    _, _, trajectory_axes = torch.pca_lowrank(encoded, q=2, center=True, niter=4)
    fit = {
        "input_mean": input_mean,
        "components": components,
        "embedding_mean": embedding_mean,
        "trajectory_axes": trajectory_axes,
        "probes": {
            name: {"mean": p.mean, "scale": p.scale, "weights": p.weights}
            for name, p in probes.items()
        },
    }
    temporary = run / "embedding-checks.partial"
    torch.save(fit, temporary)
    temporary.replace(run / "embedding-checks.pt")
    result: dict[str, object] = {
        "best_epoch": checkpoint["epoch"],
        "dimension": dimension,
        "encoder": encoder_name,
        "input_dimensions": inputs.shape[1],
        "training_samples": len(indices),
        "training_selection_seed": seed,
        "pca": "randomized PCA; 16 extra components, two power iterations",
        "linear_probe": "all signed-log change targets; standardized ridge regression",
        "ridge": ridge,
        "validation": {},
        "neighbour_rule": "cosine similarity; same cutoff; exclude the query cell",
        "query_rule": "temporal-validation cells nearest declared city coordinates",
        "limitations": "Exploratory; PCA and probes fit a fixed training subset.",
    }
    validations: dict[str, object] = {}
    for name, geographic in (("temporal", False), ("geographic", True)):
        dataset = WindowDataset(
            corpus, preprocessing, "validation", geographic=geographic
        )
        forecasts = {
            key: torch.empty((len(dataset), 6, 37), device=device, dtype=torch.float64)
            for key in probes
        }
        # Only the two small representations are retained for neighbour checks.
        retained = {
            key: torch.empty((len(dataset), dimension), device=device) for key in probes
        }
        for offset in range(0, len(dataset), 1024):
            selected = np.arange(offset, min(offset + 1024, len(dataset)))
            batch = model_inputs(model, dataset, selected, device)
            flat = (
                model.normalized_inputs(batch)
                if isinstance(model, ResidualMLP)
                else batch.flatten(1)
            )
            values = {
                encoder_name: model(batch)[1],
                "pca": (flat - input_mean) @ components,
            }
            for key, representation in values.items():
                retained[key][offset : offset + len(selected)] = representation
                forecasts[key][offset : offset + len(selected)] = (
                    probes[key].predict(representation).reshape(-1, 6, 37)
                )
        validations[name] = {
            key: score_predictions(dataset, forecast)
            for key, forecast in forecasts.items()
        }
        print(f"Completed {name} linear-probe comparison.", flush=True)
        if not geographic:
            queries = selected_cells(dataset)
            neighbours: list[dict[str, object]] = []
            # Use the final permitted cutoff consistently for all selected cells.
            candidates = np.arange(
                len(dataset.starts) - 1, len(dataset), len(dataset.starts)
            )
            for key, representation in retained.items():
                normalized = torch.nn.functional.normalize(
                    representation[candidates], dim=1
                )
                for label, cell in queries:
                    query = candidates[cell]
                    similarities = normalized @ normalized[cell]
                    similarities[cell] = -torch.inf
                    scores, order = similarities.topk(min(5, len(candidates) - 1))
                    closest = candidates[order.cpu().numpy()]
                    neighbours.append(
                        {
                            "representation": key,
                            "place": label,
                            "query": sample_keys(dataset, np.array([query])),
                            "neighbours": sample_keys(dataset, closest),
                            "query_centre": h3.cell_to_latlng(
                                dataset.corpus.split.cells[dataset.positions[cell]]
                            ),
                            "cosine_similarity": scores.cpu().tolist(),
                        }
                    )
            result["neighbours"] = neighbours
            figure, axes = plt.subplots(figsize=(8, 6), layout="constrained")
            for label, cell in queries:
                selected = np.arange(
                    cell * len(dataset.starts), (cell + 1) * len(dataset.starts)
                )
                xy = (
                    (
                        (retained[encoder_name][selected] - embedding_mean)
                        @ trajectory_axes
                    )
                    .cpu()
                    .numpy()
                )
                axes.plot(xy[:, 0], xy[:, 1], marker=".", label=f"Near {label}")
                axes.annotate("start", xy[0], fontsize=8)
                axes.annotate("end", xy[-1], fontsize=8)
            months = dataset.corpus.split.months
            first, last = (
                months[i - 1] for i in (dataset.starts[0], dataset.starts[-1])
            )
            axes.set(
                xlabel="Embedding principal component 1",
                ylabel="Embedding principal component 2",
                title=(
                    f"Validation cutoffs: {first:%b %Y}–{last:%b %Y}\n"
                    "PCA axes fitted on training embeddings"
                ),
            )
            axes.legend()
            figure.savefig(run / "trajectories.png", dpi=180)
            figure.savefig(run / "trajectories.svg")
            plt.close(figure)
        del forecasts, retained
    result["validation"] = validations
    write_json(run / "embedding-checks.json", result)
