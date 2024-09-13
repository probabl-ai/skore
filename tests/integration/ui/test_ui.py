import pytest
from fastapi.testclient import TestClient
from skore.item.item_repository import ItemRepository
from skore.layout.layout_repository import LayoutRepository
from skore.persistence.memory import InMemoryStorage
from skore.project import Project
from skore.ui.app import create_app


@pytest.fixture
def project():
    item_repository = ItemRepository(storage=InMemoryStorage())
    layout_repository = LayoutRepository(storage=InMemoryStorage())
    return Project(
        item_repository=item_repository,
        layout_repository=layout_repository,
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
    assert response.json() == {"items": {}, "layout": []}

    project.put("test", "test")
    item = project.get_item("test")

    response = client.get("/api/items")
    assert response.status_code == 200
    assert response.json() == {
        "layout": [],
        "items": {
            "test": {
                "media_type": "text/markdown",
                "value": "test",
                "updated_at": item.updated_at,
                "created_at": item.created_at,
            }
        },
    }


def test_share_report(client, project):
    project.put("test", "test")

    response = client.post("/api/report/share", json=[{"key": "test", "size": "large"}])
    assert response.status_code == 200
    assert b"<!DOCTYPE html>" in response.content


def test_put_report_layout(client):
    response = client.put("/api/report/layout", json=[{"key": "test", "size": "large"}])
    assert response.status_code == 201
