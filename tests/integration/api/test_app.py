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

        self.storage = FileSystem(directory=tmp_path)

        Store("root", storage=self.storage).insert("key", "value")
        Store("root/subroot1", storage=self.storage).insert("key", "value")
        Store("root/subroot2", storage=self.storage).insert("key", "value")
        Store("root/subroot2/subsubroot1", storage=self.storage).insert("key", "value")
        Store("root/subroot2/subsubroot2", storage=self.storage).insert(
            "key1", "value1"
        )
        Store("root/subroot2/subsubroot2", storage=self.storage).insert(
            "key2", "value2"
        )

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

        s = Store("root/subroot2/subsubroot2")
        v1, m1 = s.read("key1", metadata=True)
        v2, m2 = s.read("key2", metadata=True)
        for route in routes:
            response = client.get(route)

            assert response.status_code == 200
            assert response.json() == {
                "schema": "schema:dashboard:v0",
                "uri": "/root/subroot2/subsubroot2",
                "payload": {
                    "key1": {"type": "markdown", "data": v1, "metadata": m1},
                    "key2": {"type": "markdown", "data": v2, "metadata": m2},
                },
                "layout": [],
            }

    def test_put_layout(self, client):
        layout = [{"key": "key", "size": "small"}]
        s = Store("root", storage=self.storage)
        value, metadata = s.read("key", metadata=True)
        response = client.put(f"/api/mandrs/{s.uri}/layout", json=layout)

        assert response.status_code == 201
        assert response.json() == {
            "schema": "schema:dashboard:v0",
            "uri": "/root",
            "payload": {
                "key": {"type": "markdown", "data": value, "metadata": metadata},
            },
            "layout": layout,
        }

    def test_share(self, client):
        s = Store("root", storage=self.storage)
        response = client.get(f"/api/mandrs/share/{s.uri}")

        assert response.is_success
        assert '<script id="mandr-data" type="application/json">' in response.text
        assert response.text.count("</script>") >= 3
        assert response.text.count("</style>") >= 1
