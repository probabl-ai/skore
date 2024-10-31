import pytest
from fastapi.testclient import TestClient
from skore.ui.app import create_app
from skore.view.view import View


@pytest.fixture
def client(in_memory_project):
    return TestClient(app=create_app(project=in_memory_project))


def test_app_state(client):
    assert client.app.state.project is not None


def test_skore_ui_index(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"<!DOCTYPE html>" in response.content


def test_get_items(client, in_memory_project):
    response = client.get("/api/project/items")

    assert response.status_code == 200
    assert response.json() == {"views": {}, "items": {}}

    in_memory_project.put("test", "test")
    item = in_memory_project.get_item("test")

    response = client.get("/api/project/items")
    assert response.status_code == 200
    assert response.json() == {
        "views": {},
        "items": {
            "test": {
                "media_type": "text/markdown",
                "value": "test",
                "updated_at": item.updated_at,
                "created_at": item.created_at,
            }
        },
    }


def test_get_items_with_unserializable_objects(client, in_memory_project):
    import numpy as np

    in_memory_project.put("test", np.array([1]))

    item = in_memory_project.get_item("test")

    response = client.get("/api/project/items")
    assert response.status_code == 200
    assert response.json() == {
        "views": {},
        "items": {
            "test": {
                "media_type": "text/markdown",
                "value": "test",
                "updated_at": item.updated_at,
                "created_at": item.created_at,
            }
        },
    }


def test_put_view_layout(client):
    response = client.put("/api/project/views?key=hello", json=["test"])
    assert response.status_code == 201


def test_delete_view(client, in_memory_project):
    in_memory_project.put_view("hello", View(layout=[]))
    response = client.delete("/api/project/views?key=hello")
    assert response.status_code == 202


def test_delete_view_missing(client):
    response = client.delete("/api/project/views?key=hello")
    assert response.status_code == 404
