import os
import socket
import uuid

import psutil
import pytest
from skore.project._create import _create
from skore.project._launch import ServerInfo, _launch, cleanup_server, find_free_port


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

        err_msg = "Could not find free port after 1 attempts starting"
        with pytest.raises(RuntimeError, match=err_msg):
            find_free_port(max_attempts=1)
    finally:
        sock.close()


def test_server_info_in_memory_project(in_memory_project):
    """Check the generation of the PID file path for an in-memory project."""
    server_info = ServerInfo(in_memory_project, port=20000, pid=1234)
    assert server_info.port == 20000
    assert server_info.pid == 1234
    assert server_info.pid_file.name.startswith("skore-server-")


def test_server_info(on_disk_project):
    """Check the ServerInfo class behaviour."""
    server_info = ServerInfo(on_disk_project, port=30000, pid=1234)
    server_info.save_pid_file()
    assert server_info.pid_file.exists()
    assert server_info.load_pid_file() == {"port": 30000, "pid": 1234}

    server_info.delete_pid_file()
    assert not server_info.pid_file.exists()
    assert server_info.load_pid_file() is None


def test_launch(capsys, tmp_path):
    """Check the general behaviour of the launch function."""
    skore_project = _create(tmp_path / "test_project")
    _launch(skore_project, open_browser=False, verbose=True)
    assert "Running skore UI" in capsys.readouterr().out
    assert skore_project._server_info is not None

    server_info = skore_project._server_info
    pid_file_content = server_info.load_pid_file()
    assert server_info.port == pid_file_content["port"]
    project_identifier = uuid.uuid3(uuid.NAMESPACE_DNS, str(skore_project.path))
    assert server_info.pid_file.name == f"skore-server-{project_identifier}.json"

    skore_project.shutdown_web_ui()
    output = capsys.readouterr().out
    assert "Server that was running" in output

    _launch(skore_project, port=8000, open_browser=False, verbose=True)
    _launch(skore_project, port=8000, open_browser=False, verbose=True)
    assert "Server is already running" in capsys.readouterr().out


def test_cleanup_server_not_running(tmp_path):
    """Check that cleanup does not fail when the server is not running."""
    skore_project = _create(tmp_path / "test_project")
    cleanup_server(skore_project)


def test_cleanup_server_timeout(tmp_path, monkeypatch):
    """Test cleanup_server when process termination times out."""
    skore_project = _create(tmp_path / "test_project")

    class MockProcess:
        def __init__(self, pid):
            self.terminate_called = False
            self.kill_called = False

        def terminate(self):
            self.terminate_called = True

        def wait(self, timeout):
            raise psutil.TimeoutExpired(1, timeout)

        def kill(self):
            self.kill_called = True

    mock_process = MockProcess(1234)
    monkeypatch.setattr(psutil, "Process", lambda pid: mock_process)

    server_info = ServerInfo(skore_project, port=8000, pid=1234)
    skore_project._server_info = server_info
    server_info.save_pid_file()

    cleanup_server(skore_project)

    assert mock_process.terminate_called
    assert mock_process.kill_called
    assert not server_info.pid_file.exists()
    assert skore_project._server_info is None


def test_cleanup_server_no_process(tmp_path, monkeypatch):
    """Test cleanup_server when the process no longer exists."""
    skore_project = _create(tmp_path / "test_project")

    def mock_process_init(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", mock_process_init)

    server_info = ServerInfo(skore_project, port=8000, pid=1234)
    skore_project._server_info = server_info
    server_info.save_pid_file()

    cleanup_server(skore_project)

    assert not server_info.pid_file.exists()
    assert skore_project._server_info is None


def test_launch_zombie_process(tmp_path, monkeypatch):
    """Test launch handling when encountering a zombie process."""
    skore_project = _create(tmp_path / "test_project")
    server_info = ServerInfo(skore_project, port=8000, pid=1234)
    skore_project._server_info = server_info
    server_info.save_pid_file()

    def mock_kill(pid, signal):
        raise ProcessLookupError()

    monkeypatch.setattr(os, "kill", mock_kill)

    _launch(skore_project, port=8001, open_browser=False, verbose=True)

    assert skore_project._server_info is not None
    assert skore_project._server_info.port == 8001
    assert skore_project._server_info.pid != 1234
