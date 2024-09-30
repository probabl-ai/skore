import pytest
from fastapi.testclient import TestClient
from skore.item.item_repository import ItemRepository
from skore.persistence.in_memory_storage import InMemoryStorage
from skore.project import Project
from skore.ui.app import create_app
from skore.view.view import View
from skore.view.view_repository import ViewRepository


@pytest.fixture
def project():
    item_repository = ItemRepository(storage=InMemoryStorage())
    view_repository = ViewRepository(storage=InMemoryStorage())
    return Project(
        item_repository=item_repository,
        view_repository=view_repository,
    )


@pytest.fixture
def client(project):
    return TestClient(app=create_app(project=project))


def test_app_state(client):
    assert client.app.state.project is not None


def test_frontend_index(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"<!DOCTYPE html>" in response.content


def test_get_items(client, project):
    response = client.get("/api/items")

    assert response.status_code == 200
    assert response.json() == {"views": {}, "items": {}}

    project.put("test", "test")
    item = project.get_item("test")

    response = client.get("/api/items")
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


def test_share_view(client, project):
    project.put_view("hello", View(layout=[]))

    response = client.post("/api/report/share/hello")
    assert response.status_code == 200
    assert b"<!DOCTYPE html>" in response.content


def test_share_view_not_found(client, project):
    response = client.post("/api/report/share/hello")
    assert response.status_code == 404


def test_put_view_layout(client):
    response = client.put(
        "/api/report/view/hello",
        json=[{"key": "test", "size": "large"}],
    )
    assert response.status_code == 201


def test_delete_view(client, project):
    project.put_view("hello", View(layout=[]))
    response = client.delete("/api/report/view/hello")
    assert response.status_code == 200


def test_delete_view_missing(client):
    response = client.delete("/api/report/view/hello")
    assert response.status_code == 404
