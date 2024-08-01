from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from mandr.dashboard.app import create_dashboard_app


@pytest.fixture
def mock_now():
    return datetime.now(tz=UTC)


@pytest.fixture
def mock_nowstr(mock_now):
    return mock_now.isoformat()


@pytest.fixture
def client() -> TestClient:
    """Build the test client."""
    return TestClient(app=create_dashboard_app())
