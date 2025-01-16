import matplotlib.pyplot
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

    def test_factory_matplotlib(self, mock_nowstr):
        figure, ax = matplotlib.pyplot.subplots()

        # matplotlib.pyplot.savefig being not consistent (`xlink:href` are different
        # between two calls):
        # we can't compare figure bytes

        item = MediaItem.factory(figure)

        assert isinstance(item.media_bytes, bytes)
        assert item.media_encoding == "utf-8"
        assert item.media_type == "image/svg+xml"
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
