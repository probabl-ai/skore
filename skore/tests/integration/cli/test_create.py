import os

import pytest
from skore.cli.cli import cli
from skore.exceptions import ProjectCreationError


@pytest.fixture
def fake_working_dir(tmp_path):
    """Change directory to a temporary one, and return its path."""
    os.chdir(tmp_path)
    return tmp_path


def test_create_project_cli_default_argument(fake_working_dir):
    cli("create".split())
    assert (fake_working_dir / "project.skore").exists()


def test_create_project_cli_absolute_path(fake_working_dir):
    cli(f"create {fake_working_dir / 'hello.skore'}".split())
    assert (fake_working_dir / "hello.skore").exists()


def test_create_project_cli_ends_in_skore(fake_working_dir):
    cli("create hello.skore".split())
    assert (fake_working_dir / "hello.skore").exists()


def test_create_project_cli_invalid_name():
    with pytest.raises(ProjectCreationError):
        cli("create hello.txt".split())


def test_create_project_cli_overwrite(fake_working_dir):
    """Check the behaviour of the `overwrite` flag/parameter."""
    cli("create".split())
    assert (fake_working_dir / "project.skore").exists()

    # calling the same command without overwriting should fail
    with pytest.raises(FileExistsError):
        cli("create".split())

    # calling the same command with overwriting should succeed
    cli("create --overwrite".split())
    assert (fake_working_dir / "project.skore").exists()
