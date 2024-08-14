import pytest
from fastapi.testclient import TestClient
from mandr.api import create_api_app
from mandr.storage import FileSystem
from mandr.store import Store


class TestApiApp:
    @pytest.fixture
    def client(self):
        return TestClient(app=create_api_app())

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MANDR_ROOT", str(tmp_path))

        storage = FileSystem(directory=tmp_path)

        Store("root", storage=storage).insert("key", "value")
        Store("root/subroot1", storage=storage).insert("key", "value")
        Store("root/subroot2", storage=storage).insert("key", "value")
        Store("root/subroot2/subsubroot1", storage=storage).insert("key", "value")
        Store("root/subroot2/subsubroot2", storage=storage).insert("key1", "value1")
        Store("root/subroot2/subsubroot2", storage=storage).insert("key2", "value2")

    def test_list_stores(self, client):
        routes = [
            "/api/mandrs",
            "/api/mandrs/",
            "/api/stores",
            "/api/stores/",
        ]

        for route in routes:
            response = client.get(route)

            assert response.status_code == 200
            assert response.json() == [
                "/root",
                "/root/subroot1",
                "/root/subroot2",
                "/root/subroot2/subsubroot1",
                "/root/subroot2/subsubroot2",
            ]

    def test_get_store_by_uri(self, client):
        response = client.get("/api/mandrs/root/subroot2/subsubroot3")

        assert response.status_code == 404

        routes = [
            "/api/mandrs/root/subroot2/subsubroot2",
            "/api/stores/root/subroot2/subsubroot2",
        ]

        for route in routes:
            response = client.get(route)

            assert response.status_code == 200
            assert response.json() == {
                "schema": "schema:dashboard:v0",
                "uri": "root/subroot2/subsubroot2",
                "payload": {
                    "key1": {"type": "markdown", "data": "value1"},
                    "key2": {"type": "markdown", "data": "value2"},
                },
            }
