"""Command-line entry point for the current Atlas experiment."""

import argparse
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def doctor() -> None:
    """Print runtime versions and check tensors and gradients on the AMD GPU."""
    import torch

    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    uv_version = subprocess.run(
        ["uv", "--version"], capture_output=True, text=True, check=True, timeout=10
    )
    print(uv_version.stdout.strip())
    for package in ("atlas", "ruff", "ty", "pytest", "pre-commit"):
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "not installed"
        print(f"{package}: {installed}")

    print(f"ROCm build: {torch.version.hip or 'none'}")
    print(f"GPU available to PyTorch: {torch.cuda.is_available()}")

    if torch.version.hip is None:
        raise RuntimeError("This PyTorch installation has no ROCm support.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot access an AMD GPU. "
            "Check the driver and Compose device access."
        )
    # PyTorch uses the CUDA device API for its ROCm backend as well.
    tensor_device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(tensor_device)}")

    matrix = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], device=tensor_device, requires_grad=True
    )
    result = matrix @ matrix
    expected = torch.tensor([[7.0, 10.0], [15.0, 22.0]], device=tensor_device)
    if not torch.equal(result, expected):
        raise RuntimeError(
            f"GPU tensor operation returned an unexpected result: {result}"
        )
    print("GPU tensor check: passed")

    result.sum().backward()
    expected_gradient = torch.tensor([[7.0, 11.0], [9.0, 13.0]], device=tensor_device)
    if matrix.grad is None or not torch.equal(matrix.grad, expected_gradient):
        raise RuntimeError(
            f"GPU gradient calculation returned an unexpected result: {matrix.grad}"
        )
    print("GPU gradient check: passed")


def main() -> int:
    """Run the requested command and return its process exit status."""
    parser = argparse.ArgumentParser(
        prog="atlas", description="Learn from changes in OpenStreetMap."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "doctor", help="Show runtime versions and check AMD GPU tensors and gradients."
    )
    builder = commands.add_parser(
        "build-dataset",
        help="Build monthly features from OSM history.",
    )
    builder.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            doctor()
        else:
            from atlas.dataset import build_dataset

            build_dataset(args.config)
    except (
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        print(f"atlas {args.command} failed: {error}", file=sys.stderr)
        return 1
    return 0
