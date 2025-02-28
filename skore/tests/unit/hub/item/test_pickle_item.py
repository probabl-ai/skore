import io
import json

import pytest
from joblib import dump
from skore.hub.item import PickleItem
from skore.hub.item.item import Representation, bytes_to_b64_str


class TestPickleItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.hub.item.item.datetime", MockDatetime)

    @pytest.mark.parametrize("value", [0, 0.0, int, True, [0], {0: 0}])
    def test_factory(self, mock_nowstr, value):
        item = PickleItem.factory(value)

        with io.BytesIO() as stream:
            dump(value, stream)

            pickle_bytes = stream.getvalue()
            pickle_b64_str = bytes_to_b64_str(pickle_bytes)

        assert item.pickle_b64_str == pickle_b64_str
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr
        assert item.note is None

    def test_parameters(self, mock_nowstr):
        with io.BytesIO() as stream:
            dump(int, stream)

            pickle_bytes = stream.getvalue()
            pickle_b64_str = bytes_to_b64_str(pickle_bytes)

        item = PickleItem.factory(int)
        item_parameters = item.__parameters__

        assert item_parameters == {
            "pickle_b64_str": pickle_b64_str,
            "created_at": mock_nowstr,
            "updated_at": mock_nowstr,
            "note": None,
        }

        # Ensure parameters are JSONable
        json.dumps(item_parameters)

    def test_raw(self):
        with io.BytesIO() as stream:
            dump(int, stream)

            pickle_bytes = stream.getvalue()
            pickle_b64_str = bytes_to_b64_str(pickle_bytes)

        item1 = PickleItem.factory(int)
        item2 = PickleItem(pickle_b64_str)

        assert item1.__raw__ is int
        assert item2.__raw__ is int

    def test_representation(self):
        with io.BytesIO() as stream:
            dump(int, stream)

            pickle_bytes = stream.getvalue()
            pickle_b64_str = bytes_to_b64_str(pickle_bytes)

        representation = Representation(
            media_type="text/markdown",
            value=f"```python\n{repr(int)}\n```",
        )

        item1 = PickleItem.factory(int)
        item2 = PickleItem(pickle_b64_str)

        assert item1.__representation__ == representation
        assert item2.__representation__ == representation
