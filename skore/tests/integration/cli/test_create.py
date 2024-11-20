import subprocess

import pytest


def test_create_project_cli_default_argument(tmp_path):
    completed_process = subprocess.run(
        f"skore create --working-dir {tmp_path}".split(), capture_output=True
    )
    completed_process.check_returncode()
    assert (tmp_path / "project.skore").exists()


def test_create_project_cli_ends_in_skore(tmp_path):
    completed_process = subprocess.run(
        f"skore create hello.skore --working-dir {tmp_path}".split(),
        capture_output=True,
    )
    completed_process.check_returncode()
    assert (tmp_path / "hello.skore").exists()


def test_create_project_cli_invalid_name(tmp_path):
    completed_process = subprocess.run(
        f"skore create hello.txt --working-dir {tmp_path}".split(),
        capture_output=True,
    )
    with pytest.raises(subprocess.CalledProcessError):
        completed_process.check_returncode()
    assert b"InvalidProjectNameError" in completed_process.stderr


def test_create_project_cli_overwrite(tmp_path):
    """Check the behaviour of the `overwrite` flag/parameter."""
    completed_process = subprocess.run(
        f"skore create --working-dir {tmp_path}".split(),
        capture_output=True,
    )
    completed_process.check_returncode()
    assert (tmp_path / "project.skore").exists()

    # calling the same command without overwriting should fail
    completed_process = subprocess.run(
        f"skore create --working-dir {tmp_path}".split(),
        capture_output=True,
    )
    with pytest.raises(subprocess.CalledProcessError):
        completed_process.check_returncode()
    assert b"ProjectAlreadyExistsError" in completed_process.stderr

    # calling the same command with overwriting should succeed
    completed_process = subprocess.run(
        f"skore create --working-dir {tmp_path} --overwrite".split(),
        capture_output=True,
    )
    completed_process.check_returncode()
    assert (tmp_path / "project.skore").exists()
