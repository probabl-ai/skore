import json

import pytest
from skore.hub.item import JSONableItem
from skore.hub.item.item import ItemTypeError, Representation


class TestJSONableItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.hub.item.item.datetime", MockDatetime)

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
    def test_factory(self, mock_nowstr, value):
        item = JSONableItem.factory(value)

        assert item.value_json_str == json.dumps(value)
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr
        assert item.note is None

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            JSONableItem.factory(set())

    def test_parameters(self, mock_nowstr):
        item = JSONableItem.factory((1, 2))
        item_parameters = item.__parameters__

        assert item_parameters == {
            "value_json_str": "[1, 2]",
            "created_at": mock_nowstr,
            "updated_at": mock_nowstr,
            "note": None,
        }

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
