"""Command-line entry point for the current Atlas experiment."""

import argparse
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Literal


def doctor(device: Literal["cpu", "rocm"] = "cpu") -> None:
    """Print runtime versions and check tensors and gradients on the chosen device."""
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

    print(f"CUDA build: {torch.version.cuda or 'none'}")
    print(f"ROCm build: {torch.version.hip or 'none'}")
    print(f"CUDA/ROCm GPU available to PyTorch: {torch.cuda.is_available()}")
    print(f"MPS GPU available to PyTorch: {torch.backends.mps.is_available()}")

    if device == "rocm":
        if torch.version.hip is None:
            raise RuntimeError(
                "ROCm was requested, but this PyTorch installation has no ROCm support."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "ROCm was requested, but PyTorch cannot access an AMD GPU. "
                "Check the host driver and GPU device access."
            )
        # PyTorch uses the CUDA device API for its ROCm backend as well.
        tensor_device = torch.device("cuda:0")
        print(f"GPU: {torch.cuda.get_device_name(tensor_device)}")
    else:
        tensor_device = torch.device("cpu")

    label = device.upper()
    matrix = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], device=tensor_device, requires_grad=True
    )
    result = matrix @ matrix
    expected = torch.tensor([[7.0, 10.0], [15.0, 22.0]], device=tensor_device)
    if not torch.equal(result, expected):
        raise RuntimeError(
            f"{label} tensor operation returned an unexpected result: {result}"
        )
    print(f"{label} tensor check: passed")

    result.sum().backward()
    expected_gradient = torch.tensor([[7.0, 11.0], [9.0, 13.0]], device=tensor_device)
    if matrix.grad is None or not torch.equal(matrix.grad, expected_gradient):
        raise RuntimeError(
            f"{label} gradient calculation returned an unexpected result: {matrix.grad}"
        )
    print(f"{label} gradient check: passed")


def main() -> int:
    """Run the requested command and return its process exit status."""
    parser = argparse.ArgumentParser(
        prog="atlas", description="Learn from changes in OpenStreetMap."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    doctor_parser = commands.add_parser(
        "doctor", help="Show runtime versions and check tensors and gradients."
    )
    doctor_parser.add_argument(
        "--device",
        choices=("cpu", "rocm"),
        default="cpu",
        help="Device to check (default: cpu). ROCm requires an accessible AMD GPU.",
    )
    args = parser.parse_args()
    try:
        doctor(args.device)
    except (ImportError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"atlas doctor failed: {error}", file=sys.stderr)
        return 1
    return 0
