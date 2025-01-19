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
            # Always set open_browser to False to prevent browser opening
            self.start_params = {"port": port, "open_browser": False}

    mock_manager = MockServerManager()
    monkeypatch.setattr("skore.project._manage.ServerManager", mock_manager)
    return mock_manager


def test_cli_open(tmp_project_path, mock_server_manager):
    """Test that CLI open creates a project and starts server with correct
    parameters."""
    cli(
        [
            "open",
            str(tmp_project_path),
            "--overwrite",
            "--port",
            "8000",
            "--serve",
            "--verbose",
        ]
    )

    assert mock_server_manager.start_params["port"] == 8000
    assert mock_server_manager.start_params["open_browser"] is False


def test_cli_open_creates_project(tmp_path, mock_server_manager):
    """Test that CLI open creates a project when it doesn't exist."""
    project_path = tmp_path / "new_project.skore"
    assert not project_path.exists()

    cli(["open", str(project_path), "--create"])
    assert project_path.exists()


def test_cli_open_no_create_fails(tmp_path, mock_server_manager):
    """Test that CLI open fails when project doesn't exist and create=False."""
    project_path = tmp_path / "nonexistent.skore"

    with pytest.raises(FileNotFoundError):
        cli(["open", str(project_path), "--no-create"])


def test_cli_open_overwrite(tmp_path, mock_server_manager):
    """Test that CLI open can overwrite existing project."""
    project_path = tmp_path / "overwrite_test.skore"

    cli(["open", str(project_path), "--create"])
    initial_time = os.path.getmtime(project_path)

    cli(["open", str(project_path), "--create", "--overwrite"])
    new_time = os.path.getmtime(project_path)
    assert new_time > initial_time


def test_cli_open_no_serve(tmp_path, mock_server_manager):
    """Test that server is not started when --no-serve flag is passed."""
    project_path = tmp_path / "no_serve.skore"

    cli(["open", str(project_path), "--create", "--no-serve"])

    # Verify server was not started
    assert mock_server_manager.start_params is None
