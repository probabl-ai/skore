import os

from fastapi.testclient import TestClient
from mandr.storage import FileSystem
from mandr.store import Store


def test_index(client: TestClient):
    response = client.get("/")

    assert "html" in response.headers["Content-Type"]
    assert response.status_code == 200


def test_list_mandrs(client: TestClient, tmp_path):
    os.environ["MANDR_ROOT"] = str(tmp_path)

    storage = FileSystem(directory=tmp_path)

    Store("root", storage=storage).insert("key", "value")
    Store("root/subroot1", storage=storage).insert("key", "value")
    Store("root/subroot2", storage=storage).insert("key", "value")
    Store("root/subroot2/subsubroot1", storage=storage).insert("key", "value")
    Store("root/subroot2/subsubroot2", storage=storage).insert("key", "value")

    response = client.get("/api/mandrs")

    assert response.status_code == 200
    # NOTE: URIs always have a leading '/' in the string representation
    assert response.json() == [
        "/root",
        "/root/subroot1",
        "/root/subroot2",
        "/root/subroot2/subsubroot1",
        "/root/subroot2/subsubroot2",
    ]


def test_get_mandr(monkeypatch, client: TestClient, tmp_path):
    os.environ["MANDR_ROOT"] = str(tmp_path)

    storage = FileSystem(directory=tmp_path)

    Store("root", storage=storage).insert("key", "value")
    Store("root/subroot1", storage=storage).insert("key", "value")
    Store("root/subroot2", storage=storage).insert("key", "value")
    Store("root/subroot2/subsubroot1", storage=storage).insert("key", "value")
    Store("root/subroot2/subsubroot2", storage=storage).insert("key1", "value1")
    Store("root/subroot2/subsubroot2", storage=storage).insert("key2", "value2")

    response = client.get("/api/mandrs/root/subroot2/subsubroot2")

    assert response.status_code == 200
    assert response.json() == {
        "schema": "schema:dashboard:v0",
        "uri": "/root/subroot2/subsubroot2",
        "payload": {
            "key1": {"type": "markdown", "data": "value1"},
            "key2": {"type": "markdown", "data": "value2"},
        },
    }

    response = client.get("/api/mandrs/root/subroot2/subsubroot3")
    assert response.status_code == 404
