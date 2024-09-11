import pytest
from fastapi.testclient import TestClient
from skore.ui.app import create_app


class TestDashboardApp:
    @pytest.fixture
    def client(self):
        return TestClient(app=create_app())

    def test_frontend_index(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert b"<!DOCTYPE html>" in response.content
        assert b"<title>:skore.</title>" in response.content
