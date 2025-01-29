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


def test_cli(tmp_project_path, mock_launch):
    """Check that the CLI create a project and launch the server."""
    cli(
        [
            str(tmp_project_path),
            "--port",
            "8000",
            "--serve",
            "--verbose",
        ]
    )
    assert len(mock_launch) == 1
