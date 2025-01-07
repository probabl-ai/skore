import os

import pytest
from skore.cli.cli import cli
from skore.exceptions import ProjectCreationError


def test_create_project_cli_default_argument(tmp_path):
    os.chdir(tmp_path)
    cli("create".split())
    assert (tmp_path / "project.skore").exists()


def test_create_project_cli_absolute_path(tmp_path):
    os.chdir(tmp_path)
    cli(f"create {tmp_path / 'hello.skore'}".split())
    assert (tmp_path / "hello.skore").exists()


def test_create_project_cli_ends_in_skore(tmp_path):
    os.chdir(tmp_path)
    cli("create hello.skore".split())
    assert (tmp_path / "hello.skore").exists()


def test_create_project_cli_invalid_name():
    with pytest.raises(ProjectCreationError):
        cli("create hello.txt".split())


def test_create_project_cli_overwrite(tmp_path):
    """Check the behaviour of the `overwrite` flag/parameter."""
    os.chdir(tmp_path)
    cli("create".split())
    assert (tmp_path / "project.skore").exists()

    # calling the same command without overwriting should fail
    with pytest.raises(FileExistsError):
        cli("create".split())

    # calling the same command with overwriting should succeed
    cli("create --overwrite".split())
    assert (tmp_path / "project.skore").exists()
