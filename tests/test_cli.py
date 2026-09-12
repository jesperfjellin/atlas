"""Smoke test of the installed CLI entry point."""

import subprocess


def test_cli_help() -> None:
    result = subprocess.run(
        ["atlas", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
