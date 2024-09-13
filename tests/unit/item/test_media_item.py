import io

import altair
import matplotlib.pyplot
import PIL as pillow
import pytest
from skore.item import MediaItem


class TestMediaItem:
    def test_factory_exception(self):
        with pytest.raises(NotImplementedError):
            MediaItem.factory(None)

    def test_factory_bytes(self):
        item = MediaItem.factory(b"<content>")

        assert vars(item) == {
            "media_bytes": b"<content>",
            "media_encoding": "utf-8",
            "media_type": "application/octet-stream",
        }

    def test_factory_str(self):
        item = MediaItem.factory("<content>")

        assert vars(item) == {
            "media_bytes": b"<content>",
            "media_encoding": "utf-8",
            "media_type": "text/html",
        }

    def test_factory_altair(self):
        chart = altair.Chart().mark_point()
        chart_bytes = chart.to_json().encode("utf-8")

        item = MediaItem.factory(chart)

        assert vars(item) == {
            "media_bytes": chart_bytes,
            "media_encoding": "utf-8",
            "media_type": "application/vnd.vega.v5+json",
        }

    def test_factory_matplotlib(self, monkeypatch):
        figure, ax = matplotlib.pyplot.subplots()

        # matplotlib.pyplot.savefig being not consistent (`xlink:href` are different
        # between two calls):
        # we can't compare figure bytes

        item = MediaItem.factory(figure)

        assert isinstance(item.media_bytes, bytes)
        assert item.media_encoding == "utf-8"
        assert item.media_type == "image/svg+xml"

    def test_factory_pillow(self):
        image = pillow.Image.new("RGB", (100, 100), color="red")

        with io.BytesIO() as stream:
            image.save(stream, format="png")
            image_bytes = stream.getvalue()

        item = MediaItem.factory(image)

        assert vars(item) == {
            "media_bytes": image_bytes,
            "media_encoding": "utf-8",
            "media_type": "image/png",
        }
