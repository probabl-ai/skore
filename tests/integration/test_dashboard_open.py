import contextlib

import httpx
import pytest
from mandr.dashboard import AddressAlreadyInUseError, Dashboard


def test_dashboard_open_with_no_mandr_root(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)

    dashboard = Dashboard()
    with contextlib.closing(dashboard.open(open_browser=False)):
        assert dashboard.server.started

        host = dashboard.server.config.host
        port = dashboard.server.config.port
        response = httpx.get(f"http://{host}:{port}/api/mandrs")
        assert response.is_success


def test_dashboard_open_with_relative_mandr_root(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
    monkeypatch.setenv("MANDR_ROOT", ".datamander")

    dashboard = Dashboard()
    with contextlib.closing(dashboard.open(open_browser=False)):
        assert dashboard.server.started

        host = dashboard.server.config.host
        port = dashboard.server.config.port
        response = httpx.get(f"http://{host}:{port}/api/mandrs")
        assert response.is_success

    assert (tmp_path / ".datamander").exists()


def test_dashboard_open_with_absolute_mandr_root(monkeypatch, tmp_path):
    monkeypatch.setenv("MANDR_ROOT", str(tmp_path / ".datamander"))

    dashboard = Dashboard()
    with contextlib.closing(dashboard.open(open_browser=False)):
        assert dashboard.server.started

        host = dashboard.server.config.host
        port = dashboard.server.config.port
        response = httpx.get(f"http://{host}:{port}/api/mandrs")
        assert response.is_success

    assert (tmp_path / ".datamander").exists()


def test_dashboard_open_twice(monkeypatch, tmp_path):
    monkeypatch.setenv("MANDR_ROOT", str(tmp_path / ".datamander"))

    dashboard1 = Dashboard()
    with contextlib.closing(dashboard1.open(open_browser=False)):
        assert dashboard1.server.started

        host = dashboard1.server.config.host
        port = dashboard1.server.config.port
        response = httpx.get(f"http://{host}:{port}/api/mandrs")
        assert response.is_success

        dashboard2 = Dashboard()
        with pytest.raises(AddressAlreadyInUseError):
            dashboard2.open(open_browser=False)
