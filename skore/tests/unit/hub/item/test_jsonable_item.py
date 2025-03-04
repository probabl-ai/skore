import json

import pytest
from skore.hub.item import JSONableItem
from skore.hub.item.item import ItemTypeError, Representation


class TestJSONableItem:
    @pytest.mark.parametrize(
        "value",
        [
            0,
            1.1,
            True,
            [0, 1, 2],
            (0, 1, 2),
            {"a": 0},
            None,
        ],
    )
    def test_factory(self, value):
        item = JSONableItem.factory(value)

        assert item.value_json_str == json.dumps(value)

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            JSONableItem.factory(set())

    def test_parameters(self):
        item = JSONableItem.factory((1, 2))
        item_parameters = item.__parameters__

        assert item_parameters == {"value_json_str": "[1, 2]"}

        # Ensure parameters are JSONable
        json.dumps(item_parameters)

    def test_raw(self):
        item1 = JSONableItem.factory((1, 2))
        item2 = JSONableItem("[1, 2]")

        assert item1.__raw__ == [1, 2]
        assert item2.__raw__ == [1, 2]

    def test_representation(self):
        representation = Representation(media_type="application/json", value=[1, 2])

        item1 = JSONableItem.factory((1, 2))
        item2 = JSONableItem("[1, 2]")

        assert item1.__representation__ == representation
        assert item2.__representation__ == representation
