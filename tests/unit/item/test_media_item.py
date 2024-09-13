import io

import altair
import matplotlib.pyplot
import PIL as pillow
import pytest
from skore.item import MediaItem


class TestMediaItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.item.item.datetime", MockDatetime)

    def test_factory_exception(self):
        with pytest.raises(NotImplementedError):
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

    def test_factory_altair(self, mock_nowstr):
        chart = altair.Chart().mark_point()
        chart_bytes = chart.to_json().encode("utf-8")

        item = MediaItem.factory(chart)

        assert item.media_bytes == chart_bytes
        assert item.media_encoding == "utf-8"
        assert item.media_type == "application/vnd.vega.v5+json"
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

    def test_factory_pillow(self, mock_nowstr):
        image = pillow.Image.new("RGB", (100, 100), color="red")

        with io.BytesIO() as stream:
            image.save(stream, format="png")
            image_bytes = stream.getvalue()

        item = MediaItem.factory(image)

        assert item.media_bytes == image_bytes
        assert item.media_encoding == "utf-8"
        assert item.media_type == "image/png"
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr
