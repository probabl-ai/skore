import subprocess
from importlib.metadata import version

import pytest


def test_no_subcommand():
    """If the CLI is given no subcommand, it should output the help menu."""
    completed_process = subprocess.run("python -m skore".split())

    completed_process.check_returncode()


def test_invalid_subcommand():
    """If the CLI is given an invalid subcommand,
    it should exit and warn that the subcommand is invalid."""
    completed_process = subprocess.run(
        "python -m skore probabl-wrong-command".split(), capture_output=True
    )

    with pytest.raises(subprocess.CalledProcessError):
        completed_process.check_returncode()

    assert b"invalid" in completed_process.stderr
    assert b"probabl-wrong-command" in completed_process.stderr


def test_version():
    """The --version command should not fail."""
    completed_process = subprocess.run(
        "python -m skore --version".split(), capture_output=True
    )

    completed_process.check_returncode()
    assert f"skore {version("skore")}".encode() in completed_process.stdout
