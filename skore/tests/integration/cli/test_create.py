import pytest
from skore.cli.cli import cli
from skore.exceptions import ProjectCreationError


def test_create_project_cli_default_argument(tmp_path):
    cli(f"create --working-dir {tmp_path}".split())
    assert (tmp_path / "project.skore").exists()


def test_create_project_cli_ends_in_skore(tmp_path):
    cli(f"create hello.skore --working-dir {tmp_path}".split())
    assert (tmp_path / "hello.skore").exists()


def test_create_project_cli_invalid_name(tmp_path):
    with pytest.raises(ProjectCreationError):
        cli(f"create hello.txt --working-dir {tmp_path}".split())


def test_create_project_cli_overwrite(tmp_path):
    """Check the behaviour of the `overwrite` flag/parameter."""
    cli(f"create --working-dir {tmp_path}".split())
    assert (tmp_path / "project.skore").exists()

    # calling the same command without overwriting should fail
    with pytest.raises(FileExistsError):
        cli(f"create --working-dir {tmp_path}".split())

    # calling the same command with overwriting should succeed
    cli(f"create --working-dir {tmp_path} --overwrite".split())
    assert (tmp_path / "project.skore").exists()
