from io import BytesIO
from json import dumps

import PIL.Image
from pytest import raises
from skore_hub_project.item import PillowImageItem
from skore_hub_project.item.item import ItemTypeError, bytes_to_b64_str


class TestPillowImageItem:
    def test_factory(self):
        image = PIL.Image.new("RGB", (100, 100), color="red")
        image_bytes = image.tobytes()
        image_b64_str = bytes_to_b64_str(image_bytes)

        item = PillowImageItem.factory(image)

        assert item.image_b64_str == image_b64_str
        assert item.image_mode == image.mode
        assert item.image_size == image.size

    def test_factory_exception(self):
        with raises(ItemTypeError):
            PillowImageItem.factory(None)

    def test_parameters(self):
        image = PIL.Image.new("RGB", (100, 100), color="red")
        image_bytes = image.tobytes()
        image_b64_str = bytes_to_b64_str(image_bytes)

        item = PillowImageItem.factory(image)
        item_parameters = item.__parameters__

        assert item_parameters == {
            "parameters": {
                "class": "PillowImageItem",
                "parameters": {
                    "image_b64_str": image_b64_str,
                    "image_mode": image.mode,
                    "image_size": image.size,
                },
            }
        }

        # Ensure parameters are JSONable
        dumps(item_parameters)

    def test_raw(self):
        image = PIL.Image.new("RGB", (100, 100), color="red")
        image_bytes = image.tobytes()
        image_b64_str = bytes_to_b64_str(image_bytes)

        item1 = PillowImageItem.factory(image)
        item2 = PillowImageItem(
            image_b64_str=image_b64_str,
            image_mode=image.mode,
            image_size=image.size,
        )

        assert item1.__raw__ == image
        assert item2.__raw__ == image

    def test_representation(self):
        image = PIL.Image.new("RGB", (100, 100), color="red")
        image_bytes = image.tobytes()
        image_b64_str = bytes_to_b64_str(image_bytes)

        with BytesIO() as stream:
            image.save(stream, format="png")

            png_bytes = stream.getvalue()
            png_b64_str = bytes_to_b64_str(png_bytes)
            representation = {
                "representation": {
                    "media_type": "image/png;base64",
                    "value": png_b64_str,
                }
            }

        item1 = PillowImageItem.factory(image)
        item2 = PillowImageItem(
            image_b64_str=image_b64_str,
            image_mode=image.mode,
            image_size=image.size,
        )

        assert item1.__representation__ == representation
        assert item2.__representation__ == representation

        # Ensure representation is JSONable
        dumps(item1.__representation__)
        dumps(item2.__representation__)
