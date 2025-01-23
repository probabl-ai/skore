import socket

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
    _launch(skore_project, port=8000, open_browser=False, verbose=True)

    assert skore_project._server_manager is not None
    server_manager = skore_project._server_manager
    assert server_manager is ServerManager.get_instance()
    assert "Running skore UI" in capsys.readouterr().out

    # Force server shutdown
    server_manager._server_running = True  # ensure it's marked as running
    server_manager.shutdown()
    assert server_manager._server_running is False

    # Check shutdown output
    output = capsys.readouterr().out
    assert "Server that was running" in output
    assert not server_manager._server_running
    assert server_manager._loop is None

    # # Try launching again
    _launch(skore_project, port=8000, open_browser=False, verbose=True)
    _launch(skore_project, port=8000, open_browser=False, verbose=True)
    assert skore_project._server_manager is ServerManager.get_instance()
    assert "Server is already running" in capsys.readouterr().out
