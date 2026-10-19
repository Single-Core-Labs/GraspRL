"""Smoke test for package installation and import."""

import subprocess
import sys


def test_package_import():
    """Test that the package can be imported."""
    import grasp_rl
    assert grasp_rl.__version__ == "0.1.0"


def test_cli_help():
    """Test that the CLI help command works."""
    result = subprocess.run(
        [sys.executable, "-m", "grasp_rl.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "GraspRL" in result.stdout
