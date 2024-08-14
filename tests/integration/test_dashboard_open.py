import contextlib

import httpx
from mandr.dashboard import Dashboard


def test_dashboard_open_with_no_mandr_root(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)

    dashboard = Dashboard()
    with contextlib.closing(dashboard.open(open_browser=False)):
        assert dashboard.server.started

        host = dashboard.server.config.host
        port = dashboard.server.config.port
        response = httpx.get(f"http://{host}:{port}/api/mandrs")
        assert response.is_success


def test_dashboard_open_with_relative_mandr_root(monkeypatch):
    monkeypatch.setenv("MANDR_ROOT", ".datamander")

    dashboard = Dashboard()
    with contextlib.closing(dashboard.open(open_browser=False)):
        assert dashboard.server.started

        host = dashboard.server.config.host
        port = dashboard.server.config.port
        response = httpx.get(f"http://{host}:{port}/api/mandrs")
        assert response.is_server_error


def test_dashboard_open_with_absolute_mandr_root(monkeypatch, tmp_path):
    monkeypatch.setenv("MANDR_ROOT", str(tmp_path))

    dashboard = Dashboard()
    with contextlib.closing(dashboard.open(open_browser=False)):
        assert dashboard.server.started

        host = dashboard.server.config.host
        port = dashboard.server.config.port
        response = httpx.get(f"http://{host}:{port}/api/mandrs")
        assert response.is_success
