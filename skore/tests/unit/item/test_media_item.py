import pytest
from skore.persistence.item import ItemTypeError, MediaItem


class TestMediaItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            MediaItem.factory(None)

    def test_factory_bytes(self, mock_nowstr):
        item = MediaItem.factory(b"<content>")

        assert item.media_bytes == b"<content>"
        assert item.media_encoding == "utf-8"
        assert item.media_type == "application/octet-stream"
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    def test_factory_str(self, mock_nowstr):
        item = MediaItem.factory("<content>")

        assert item.media_bytes == b"<content>"
        assert item.media_encoding == "utf-8"
        assert item.media_type == "text/markdown"
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    def test_get_serializable_dict(self, mock_nowstr):
        item = MediaItem.factory("<content>")

        serializable = item.as_serializable_dict()
        assert serializable == {
            "updated_at": mock_nowstr,
            "created_at": mock_nowstr,
            "note": None,
            "media_type": "text/markdown",
            "value": "<content>",
        }
