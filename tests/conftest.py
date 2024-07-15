from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from mandr import InfoMander
from mandr.dashboard.webapp import app


@pytest.fixture
def mock_now():
    return datetime.now(tz=UTC)


@pytest.fixture
def mock_nowstr(mock_now):
    return mock_now.isoformat()


@pytest.fixture
def mock_mandr(monkeypatch, mock_now, tmp_path):
    class MockCache(dict):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def iterkeys(self, *args, **kwargs):
            yield from self.keys()

    class MockDatetime:
        @staticmethod
        def now(*args, **kwargs):
            return mock_now

    monkeypatch.setattr("mandr.infomander.Cache", MockCache)
    monkeypatch.setattr("mandr.infomander.datetime", MockDatetime)

    return InfoMander("root", root=tmp_path)


@pytest.fixture
def client() -> TestClient:
    """Build the test client."""
    return TestClient(app=app)
