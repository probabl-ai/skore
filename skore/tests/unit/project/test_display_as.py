import pytest
from skore.persistence.item import MediaItem, MediaType


@pytest.fixture(autouse=True)
def monkeypatch_datetime(monkeypatch, MockDatetime):
    monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)


def test_str_without_display_as(in_memory_project, mock_nowstr):
    in_memory_project.put("key", "<str>")

    item = in_memory_project._item_repository.get_item("key")

    assert isinstance(item, MediaItem)
    assert item.media == "<str>"
    assert item.media_type == "text/markdown"


@pytest.mark.parametrize("media_type", list(MediaType))
def test_str_with_display_as(in_memory_project, mock_nowstr, media_type):
    in_memory_project.put("key", "<str>", display_as=media_type.name)

    item = in_memory_project._item_repository.get_item("key")

    assert isinstance(item, MediaItem)
    assert item.media == "<str>"
    assert item.media_type == media_type.value


def test_exception(in_memory_project, mock_nowstr):
    with pytest.raises(TypeError):
        in_memory_project.put("key", 1, display_as="MARKDOWN")

    with pytest.raises(ValueError):
        in_memory_project.put("key", "<str>", display_as="<display_as>")
