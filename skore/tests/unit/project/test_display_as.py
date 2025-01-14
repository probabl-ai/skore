import pytest
from skore.persistence.item import MediaItem, PrimitiveItem


@pytest.fixture(autouse=True)
def monkeypatch_datetime(monkeypatch, MockDatetime):
    monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)


def test_str_without_display_as(in_memory_project, mock_nowstr):
    in_memory_project.put("key", "<str>")

    item = in_memory_project.item_repository.get_item("key")

    assert isinstance(item, PrimitiveItem)
    assert item.primitive == "<str>"


def test_str_with_display_as(in_memory_project, mock_nowstr):
    in_memory_project.put("key", "<str>", display_as="MARKDOWN")

    item = in_memory_project.item_repository.get_item("key")

    assert isinstance(item, MediaItem)
    assert item.media_bytes == b"<str>"
    assert item.media_encoding == "utf-8"
    assert item.media_type == "text/markdown"


def test_exception(in_memory_project, mock_nowstr):
    with pytest.raises(TypeError):
        in_memory_project.put("key", 1, display_as="MARKDOWN")

    with pytest.raises(ValueError):
        in_memory_project.put("key", "<str>", display_as="<display_as>")
