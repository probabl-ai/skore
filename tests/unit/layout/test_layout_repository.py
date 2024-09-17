import pytest
from skore.layout import LayoutRepository
from skore.layout.layout import LayoutItem, LayoutItemSize
from skore.persistence.in_memory_storage import InMemoryStorage


@pytest.fixture
def layout_repository():
    return LayoutRepository(InMemoryStorage())


def test_get(layout_repository):
    layout = [
        LayoutItem(key="key1", size=LayoutItemSize.LARGE),
        LayoutItem(key="key2", size=LayoutItemSize.SMALL),
    ]

    layout_repository.put_layout(layout)

    assert layout_repository.get_layout() == layout


def test_get_with_no_put(layout_repository):
    with pytest.raises(KeyError):
        layout_repository.get_layout()


def test_delete(layout_repository):
    layout_repository.put_layout([])

    layout_repository.delete_layout()

    with pytest.raises(KeyError):
        layout_repository.get_layout()


def test_delete_with_no_put(layout_repository):
    with pytest.raises(KeyError):
        layout_repository.delete_layout()
