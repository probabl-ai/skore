from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mandr.dashboard.webapp import app


@pytest.fixture
def client() -> TestClient:
    """Build the test client."""
    return TestClient(app=app)


@pytest.fixture(autouse=True)
def create_manders_in_tmp(monkeypatch, tmp_path):
    """Create all manders in a tmp path for test purposes."""

    def mocked_get_storage_path() -> Path:
        """Return a path to the local mander storage."""
        return Path(tmp_path)

    monkeypatch.setattr("mandr.infomander._get_storage_path", mocked_get_storage_path)
