import contextlib
import threading
import time

import httpx
import pytest
import uvicorn


class ThreadableServer(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        start = time.monotonic()
        try:
            while not self.started:
                time.sleep(1e-3)
                if time.monotonic() - start > 1:
                    break
            yield
        finally:
            self.should_exit = True
            thread.join()


config = uvicorn.Config(
    app="mandr.dashboard.app:create_dashboard_app",
    port=22140,
    log_level="error",
    factory=True,
)


def test_dashboard_open_with_no_mandr_root(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)

    with ThreadableServer(config=config).run_in_thread():
        host = config.host
        port = config.port
        response = httpx.get(f"http://{host}:{port}/api/mandrs")
        assert response.is_success


def test_dashboard_open_with_relative_mandr_root(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
    monkeypatch.setenv("MANDR_ROOT", ".datamander")

    with ThreadableServer(config=config).run_in_thread():
        host = config.host
        port = config.port
        response = httpx.get(f"http://{host}:{port}/api/mandrs")
        assert response.is_success

    assert (tmp_path / ".datamander").exists()


def test_dashboard_open_with_absolute_mandr_root(monkeypatch, tmp_path):
    monkeypatch.setenv("MANDR_ROOT", str(tmp_path / ".datamander"))

    with ThreadableServer(config=config).run_in_thread():
        host = config.host
        port = config.port
        response = httpx.get(f"http://{host}:{port}/api/mandrs")
        assert response.is_success

    assert (tmp_path / ".datamander").exists()


def test_dashboard_open_twice(monkeypatch, tmp_path, caplog):
    """Running the app twice at the same port fails because the address
    is already in use when the second app is ran."""
    monkeypatch.setenv("MANDR_ROOT", str(tmp_path / ".datamander"))

    with ThreadableServer(config=config).run_in_thread():
        host = config.host
        port = config.port
        response = httpx.get(f"http://{host}:{port}/api/mandrs")
        assert response.is_success

        with pytest.raises(SystemExit):
            # TODO: assert "Address is already in use" was properly logged
            # (caplog does not record anything)
            uvicorn.Server(config=config).run()
