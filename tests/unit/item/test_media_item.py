import pytest
from skore.persistence.item import ItemTypeError, MediaItem, MediaType


class TestMediaItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            MediaItem.factory(None)

        with pytest.raises(ValueError):
            MediaItem.factory("<content>", "application/octet-stream")

    @pytest.mark.parametrize("media_type", [enum.value for enum in MediaType])
    def test_factory(self, mock_nowstr, media_type):
        item = MediaItem.factory("<content>", media_type)

        assert item.media == "<content>"
        assert item.media_type == media_type
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr
