import pytest
from skore.persistence.repository import ViewRepository
from skore.persistence.storage import InMemoryStorage
from skore.persistence.view.view import View


@pytest.fixture
def view_repository():
    return ViewRepository(InMemoryStorage())


def test_get(view_repository):
    view = View(layout=["key1", "key2"])

    view_repository.put_view("view", view)

    assert view_repository.get_view("view") == view


def test_get_with_no_put(view_repository):
    with pytest.raises(KeyError):
        view_repository.get_view("view")


def test_delete(view_repository):
    view_repository.put_view("view", View(layout=[]))

    view_repository.delete_view("view")

    with pytest.raises(KeyError):
        view_repository.get_view("view")


def test_delete_with_no_put(view_repository):
    with pytest.raises(KeyError):
        view_repository.delete_view("view")
