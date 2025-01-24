"""Test CLI properly calls the app."""

import os

import pytest
from skore.cli.cli import cli


@pytest.fixture
def tmp_project_path(tmp_path):
    """Create a project at `tmp_path` and return its absolute path."""
    # Project path must end with ".skore"
    project_path = tmp_path.parent / (tmp_path.name + ".skore")
    os.mkdir(project_path)
    os.mkdir(project_path / "items")
    os.mkdir(project_path / "views")
    return project_path


@pytest.fixture
def mock_launch(monkeypatch):
    """Fixture to patch _launch function to prevent browser opening."""

    def mock_launch_fn(project, port=None, open_browser=True, verbose=False):
        from skore.project._launch import _launch as original_launch

        return original_launch(project, port=port, open_browser=False, verbose=verbose)

    monkeypatch.setattr("skore.cli.cli._launch", mock_launch_fn)


def test_cli_launch(tmp_project_path):
    """Test that CLI launch starts server with correct parameters."""
    cli(
        [
            "launch",
            str(tmp_project_path),
            "--port",
            "8000",
            "--no-open-browser",
            "--verbose",
        ]
    )


def test_cli_launch_no_project_name():
    with pytest.raises(SystemExit):
        cli(["launch", "--port", 0, "--no-open-browser", "--verbose"])


def test_cli_create(tmp_path):
    """Test that CLI create command creates the expected directory structure."""
    os.chdir(tmp_path)  # Change to temp directory for relative path testing

    cli(["create", "test_project", "--verbose"])

    project_path = tmp_path / "test_project.skore"
    assert project_path.exists()
    assert (project_path / "items").exists()
    assert (project_path / "views").exists()

    with pytest.raises(FileExistsError):
        cli(["create", "test_project"])

    cli(["create", "test_project", "--overwrite"])
    assert project_path.exists()


def test_cli_open(tmp_path, mock_launch):
    """Test that CLI open command works with different scenarios."""
    os.chdir(tmp_path)

    cli(["open", "test_project", "--no-serve"])
    project_path = tmp_path / "test_project.skore"
    assert project_path.exists()
    assert (project_path / "items").exists()
    assert (project_path / "views").exists()

    cli(["open", "test_project", "--port", "8000", "--verbose"])
    cli(["open", "test_project", "--no-create", "--port", "8001"])

    with pytest.raises(FileNotFoundError):
        cli(["open", "nonexistent_project", "--no-create"])
