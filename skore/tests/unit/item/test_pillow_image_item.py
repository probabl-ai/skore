import io
import json

import PIL.Image
import pytest
from skore.persistence.item import ItemTypeError, PillowImageItem
from skore.utils import bytes_to_b64_str


class TestPillowImageItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)

    def test_factory(self, mock_nowstr):
        image = PIL.Image.new("RGB", (100, 100), color="red")
        image_bytes = image.tobytes()
        image_b64_str = bytes_to_b64_str(image_bytes)

        item = PillowImageItem.factory(image)

        assert item.image_b64_str == image_b64_str
        assert item.image_mode == image.mode
        assert item.image_size == image.size
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            PillowImageItem.factory(None)

    def test_ensure_jsonable(self):
        image = PIL.Image.new("RGB", (100, 100), color="red")

        item = PillowImageItem.factory(image)
        item_parameters = item.__parameters__

        json.dumps(item_parameters)

    def test_image(self):
        image = PIL.Image.new("RGB", (100, 100), color="red")
        image_bytes = image.tobytes()
        image_b64_str = bytes_to_b64_str(image_bytes)

        item1 = PillowImageItem.factory(image)
        item2 = PillowImageItem(
            image_b64_str=image_b64_str,
            image_mode=image.mode,
            image_size=image.size,
        )

        assert item1.image == image
        assert item2.image == image

    def test_as_serializable_dict(self, mock_nowstr):
        image = PIL.Image.new("RGB", (100, 100), color="red")
        item = PillowImageItem.factory(image)

        with io.BytesIO() as stream:
            image.save(stream, format="png")

            png_bytes = stream.getvalue()
            png_b64_str = bytes_to_b64_str(png_bytes)

        assert item.as_serializable_dict() == {
            "updated_at": mock_nowstr,
            "created_at": mock_nowstr,
            "note": None,
            "media_type": "image/png;base64",
            "value": png_b64_str,
        }
