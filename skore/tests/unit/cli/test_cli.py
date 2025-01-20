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
def mock_server_manager(monkeypatch):
    """Mock ServerManager to verify server startup parameters."""

    class MockServerManager:
        def __init__(self):
            self.start_params = None

        def get_instance(self):
            return self

        def start_server(self, project, port=None, open_browser=True):
            self.start_params = {"port": port, "open_browser": open_browser}

    mock_manager = MockServerManager()
    monkeypatch.setattr("skore.project._launch.ServerManager", mock_manager)
    return mock_manager


def test_cli_launch(tmp_project_path, mock_server_manager):
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

    assert mock_server_manager.start_params["port"] == 8000
    assert mock_server_manager.start_params["open_browser"] is False


def test_cli_launch_no_project_name():
    with pytest.raises(SystemExit):
        cli(["launch", "--port", 0, "--no-open-browser", "--verbose"])


def test_cli_create(tmp_path, monkeypatch):
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


def test_cli_open(tmp_path, mock_server_manager):
    """Test that CLI open command works with different scenarios."""
    os.chdir(tmp_path)

    cli(["open", "test_project", "--no-serve"])
    project_path = tmp_path / "test_project.skore"
    assert project_path.exists()
    assert (project_path / "items").exists()
    assert (project_path / "views").exists()
    assert mock_server_manager.start_params is None

    cli(["open", "test_project", "--port", "8000", "--verbose"])
    assert mock_server_manager.start_params["port"] == 8000

    cli(["open", "test_project", "--no-create", "--port", "8001"])
    assert mock_server_manager.start_params["port"] == 8001

    with pytest.raises(FileNotFoundError):
        cli(["open", "nonexistent_project", "--no-create"])
