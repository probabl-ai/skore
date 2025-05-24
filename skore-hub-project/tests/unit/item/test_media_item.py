from json import dumps

import pytest
from skore_hub_project.item import MediaItem, MediaType
from skore_hub_project.item.item import ItemTypeError


class TestMediaItem:
    @pytest.mark.parametrize("media_type", [enum.value for enum in MediaType])
    def test_factory(self, media_type):
        item = MediaItem.factory("<content>", media_type)

        assert item.media == "<content>"
        assert item.media_type == media_type

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            MediaItem.factory(None)

        with pytest.raises(ValueError):
            MediaItem.factory("<content>", "application/octet-stream")

    @pytest.mark.parametrize("media_type", [enum.value for enum in MediaType])
    def test_parameters(self, media_type):
        item = MediaItem.factory("<content>", media_type)
        item_parameters = item.__parameters__

        assert item_parameters == {
            "parameters": {
                "class": "MediaItem",
                "parameters": {"media": "<content>", "media_type": media_type},
            }
        }

        # Ensure parameters are JSONable
        dumps(item_parameters)

    def test_raw(self):
        item1 = MediaItem.factory("<content>")
        item2 = MediaItem("<content>", MediaType.MARKDOWN.value)

        assert item1.__raw__ == "<content>"
        assert item2.__raw__ == "<content>"

    def test_representation(self):
        representation = {
            "representation": {
                "media_type": "text/markdown",
                "value": "<content>",
            }
        }

        item1 = MediaItem.factory("<content>")
        item2 = MediaItem("<content>", MediaType.MARKDOWN.value)

        assert item1.__representation__ == representation
        assert item2.__representation__ == representation

        # Ensure representation is JSONable
        dumps(item1.__representation__)
        dumps(item2.__representation__)
