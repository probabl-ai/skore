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
    """Fixture that mocks the _launch function and tracks calls to it."""
    calls = []

    def _mock_launch(
        project, keep_alive="auto", port=None, open_browser=True, verbose=False
    ):
        calls.append((project, keep_alive, port, open_browser, verbose))

    monkeypatch.setattr("skore.project._launch._launch", _mock_launch)
    return calls


def test_cli_open(tmp_project_path, mock_launch):
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
    assert len(mock_launch) == 1


def test_cli_open_creates_project(tmp_path, mock_launch):
    """Test that CLI open creates a project when it doesn't exist."""
    project_path = tmp_path / "new_project.skore"
    assert not project_path.exists()

    cli(["open", str(project_path), "--create"])
    assert project_path.exists()
    assert len(mock_launch) == 1


def test_cli_open_no_create_fails(tmp_path, mock_launch):
    """Test that CLI open fails when project doesn't exist and create=False."""
    project_path = tmp_path / "nonexistent.skore"

    with pytest.raises(FileNotFoundError):
        cli(["open", str(project_path), "--no-create"])
    assert len(mock_launch) == 0


def test_cli_open_overwrite(tmp_path, mock_launch):
    """Test that CLI open can overwrite existing project."""
    project_path = tmp_path / "overwrite_test.skore"

    cli(["open", str(project_path), "--create"])
    initial_time = os.path.getmtime(project_path)

    cli(["open", str(project_path), "--create", "--overwrite"])
    new_time = os.path.getmtime(project_path)
    assert new_time > initial_time
    assert len(mock_launch) == 2


def test_cli_open_no_serve(tmp_path, mock_launch):
    """Test that server is not started when --no-serve flag is passed."""
    project_path = tmp_path / "no_serve.skore"

    cli(["open", str(project_path), "--create", "--no-serve"])
    assert len(mock_launch) == 0
