from unittest.mock import MagicMock, patch

import pytest
import skore
from skore.project._launch import find_free_port
from skore.ui.server import run_server


@pytest.fixture
def mock_uvicorn():
    """Fixture to mock uvicorn components and capture the server instance."""
    server_instance = MagicMock()
    mock_config = MagicMock()

    with (
        patch("uvicorn.Server", return_value=server_instance) as mock_server,
        patch("uvicorn.Config", return_value=mock_config) as mock_config_class,
    ):
        yield {
            "server": mock_server,
            "server_instance": server_instance,
            "config": mock_config_class,
            "config_instance": mock_config,
        }


def test_run_server(mock_uvicorn, tmp_path):
    """Test that run_server creates and runs a server with correct configuration."""
    project_dir = tmp_path / "test_project"
    skore.open(project_dir, serve=False)

    port = find_free_port()
    run_server(port=port, project_path=str(project_dir))

    mock_uvicorn["config"].assert_called_once()
    config_kwargs = mock_uvicorn["config"].call_args[1]
    assert config_kwargs["port"] == port
    assert config_kwargs["host"] == "127.0.0.1"
    assert config_kwargs["log_level"] == "error"

    mock_uvicorn["server"].assert_called_once_with(mock_uvicorn["config_instance"])
    mock_uvicorn["server_instance"].run.assert_called_once()
