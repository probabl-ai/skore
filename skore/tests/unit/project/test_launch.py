import socket
import time

from skore.project._create import _create
from skore.project._launch import (
    ServerManager,
    _launch,
    find_free_port,
)


def test_find_free_port():
    """Test that find_free_port returns a valid port number"""
    port = find_free_port()
    assert isinstance(port, int)
    assert port > 0

    # Verify we can bind to the port since it should be free
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", port))
        sock.listen(1)
    finally:
        sock.close()


def test_server_manager_singleton():
    """Test ServerManager singleton pattern"""
    server_manager = ServerManager.get_instance()
    assert isinstance(server_manager, ServerManager)
    assert ServerManager.get_instance() is server_manager


def test_launch(capsys, tmp_path):
    """Check the general behaviour of the launch function."""
    skore_project = _create(tmp_path / "test_project")
    try:
        _launch(skore_project, port=8000, open_browser=False, verbose=True)

        time.sleep(0.1)  # let the server start
        assert skore_project._server_manager is not None
        assert skore_project._server_manager is ServerManager.get_instance()
        assert "Running skore UI" in capsys.readouterr().out

        _launch(skore_project, port=8000, open_browser=False, verbose=True)

        time.sleep(0.1)
        assert skore_project._server_manager is ServerManager.get_instance()
        assert "Server is already running" in capsys.readouterr().out
    finally:
        skore_project.shutdown_web_ui()
        time.sleep(0.1)  # let the server shutdown
        assert "Server that was running" in capsys.readouterr().out
