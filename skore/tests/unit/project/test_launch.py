import socket

from skore.project._create import _create
from skore.project._launch import _launch, find_free_port


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


def test_launch(capsys, tmp_path):
    """Check the general behaviour of the launch function."""
    skore_project = _create(tmp_path / "test_project")
    _launch(skore_project, port=8000, open_browser=False, verbose=True)
    assert "Running skore UI" in capsys.readouterr().out

    skore_project.shutdown_web_ui()
    output = capsys.readouterr().out
    assert "Server that was running" in output

    _launch(skore_project, port=8000, open_browser=False, verbose=True)
    _launch(skore_project, port=8000, open_browser=False, verbose=True)
    assert "Server is already running" in capsys.readouterr().out
