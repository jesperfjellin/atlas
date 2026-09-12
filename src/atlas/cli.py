"""Command-line entry point for the current Atlas experiment."""

import argparse
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version


def doctor() -> None:
    """Print runtime versions and check a small CPU tensor operation."""
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

    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu")
    result = matrix @ matrix
    expected = torch.tensor([[7.0, 10.0], [15.0, 22.0]], device="cpu")
    if not torch.equal(result, expected):
        raise RuntimeError(
            f"CPU tensor operation returned an unexpected result: {result}"
        )
    print("CPU tensor check: passed")


def main() -> int:
    """Run the requested command and return its process exit status."""
    parser = argparse.ArgumentParser(
        prog="atlas", description="Learn from changes in OpenStreetMap."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Show runtime versions and check CPU tensors.")
    parser.parse_args()
    try:
        doctor()
    except (ImportError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"atlas doctor failed: {error}", file=sys.stderr)
        return 1
    return 0
