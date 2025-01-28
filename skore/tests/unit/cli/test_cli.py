"""Test CLI properly calls the app."""

import os

import pytest
from skore import open
from skore.cli.cli import cli
from skore.project._launch import ServerInfo


@pytest.fixture
def tmp_project_path(tmp_path):
    """Create a project at `tmp_path` and return its absolute path."""
    # Project path must end with ".skore"
    project_path = tmp_path.parent / (tmp_path.name + ".skore")
    os.mkdir(project_path)
    os.mkdir(project_path / "items")
    os.mkdir(project_path / "views")
    return project_path


def close_project(path):
    """Force closing the web UI."""
    from skore import open

    project = open(path, serve=False)
    assert project._server_info is not None
    pid_file = project._server_info.pid_file
    assert pid_file.exists()
    project.shutdown_web_ui()
    assert not pid_file.exists()


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


def test_cli_open(tmp_path, monkeypatch):
    """Test that CLI open command works with different scenarios."""
    # Force open_browser to False for all _launch calls
    from skore.project._launch import _launch

    monkeypatch.setattr(
        "skore.project._launch._launch",
        lambda *args, **kwargs: _launch(*args, **{**kwargs, "open_browser": False}),
    )

    project_path = tmp_path / "project_cli_open.skore"

    cli(["open", str(project_path), "--no-serve"])
    assert project_path.exists()
    assert (project_path / "items").exists()
    assert (project_path / "views").exists()

    cli(["open", str(project_path), "--verbose"])
    close_project(project_path)
    cli(["open", str(project_path), "--no-create"])
    close_project(project_path)

    with pytest.raises(FileNotFoundError):
        cli(["open", "nonexistent_project", "--no-create"])


def test_cli_kill(tmp_project_path, monkeypatch):
    """Test that CLI kill command properly terminates all running servers."""
    # Force open_browser to False for all _launch calls
    from skore.project._launch import _launch

    monkeypatch.setattr(
        "skore.project._launch._launch",
        lambda *args, **kwargs: _launch(*args, **{**kwargs, "open_browser": False}),
    )

    project = open(tmp_project_path, serve=True, keep_alive=False)
    pid_file = ServerInfo._get_pid_file_path(project)
    assert pid_file.exists()

    cli(["kill", "--verbose"])
    assert not pid_file.exists()


def test_cli_kill_no_servers():
    """Test that CLI kill command works safely when no servers are running."""
    cli(["kill", "--verbose"])
