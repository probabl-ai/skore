import pytest
from fastapi.testclient import TestClient
from skore.persistence.memory import InMemoryStorage
from skore.project import Project
from skore.ui.app import create_app


@pytest.fixture
def storage():
    return InMemoryStorage()


@pytest.fixture
def project(storage):
    return Project(storage=storage)


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
    assert response.json() == {}

    project.put("test", "test")
    response = client.get("/api/items")
    assert response.status_code == 200
    assert response.json() == {
        "test": {"item_type": "json", "media_type": None, "serialized": "test"}
    }


def test_share_report(client, project):
    project.put("test", "test")

    response = client.post("/api/report/share", json=[{"key": "test", "size": "large"}])
    assert response.status_code == 200
    assert b"<!DOCTYPE html>" in response.content
