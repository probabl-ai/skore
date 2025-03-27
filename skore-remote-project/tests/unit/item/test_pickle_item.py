import io
import json

import pytest
from joblib import dump
from skore_remote_project.item import PickleItem
from skore_remote_project.item.item import bytes_to_b64_str


class TestPickleItem:
    @pytest.mark.parametrize("value", [0, 0.0, int, True, [0], {0: 0}])
    def test_factory(self, value):
        item = PickleItem.factory(value)

        with io.BytesIO() as stream:
            dump(value, stream)

            pickle_bytes = stream.getvalue()
            pickle_b64_str = bytes_to_b64_str(pickle_bytes)

        assert item.pickle_b64_str == pickle_b64_str

    def test_parameters(self):
        with io.BytesIO() as stream:
            dump(int, stream)

            pickle_bytes = stream.getvalue()
            pickle_b64_str = bytes_to_b64_str(pickle_bytes)

        item = PickleItem.factory(int)
        item_parameters = item.__parameters__

        assert item_parameters == {
            "parameters": {
                "class": "PickleItem",
                "parameters": {"pickle_b64_str": pickle_b64_str},
            }
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

        representation = {
            "representation": {
                "media_type": "text/markdown",
                "value": f"```python\n{repr(int)}\n```",
            }
        }

        item1 = PickleItem.factory(int)
        item2 = PickleItem(pickle_b64_str)

        assert item1.__representation__ == representation
        assert item2.__representation__ == representation

        # Ensure representation is JSONable
        json.dumps(item1.__representation__)
        json.dumps(item2.__representation__)
