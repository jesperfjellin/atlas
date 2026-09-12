"""Smoke test of the installed CLI and the required CPU runtime."""

import subprocess


def test_doctor_runs_on_cpu() -> None:
    result = subprocess.run(
        ["atlas", "doctor"], capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0, result.stdout + result.stderr
