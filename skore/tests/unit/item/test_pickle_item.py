import io
import json

import joblib
import pytest
from skore.persistence.item import PickleItem
from skore.utils import bytes_to_b64_str


class TestPickleItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)

    @pytest.mark.parametrize("object", [0, 0.0, int, True, [0], {0: 0}])
    def test_factory(self, mock_nowstr, object):
        item = PickleItem.factory(object)

        with io.BytesIO() as stream:
            joblib.dump(object, stream)

            pickle_bytes = stream.getvalue()
            pickle_b64_str = bytes_to_b64_str(pickle_bytes)

        assert item.pickle_b64_str == pickle_b64_str
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    def test_ensure_jsonable(self):
        item = PickleItem.factory(object)
        item_parameters = item.__parameters__

        json.dumps(item_parameters)

    def test_object(self, mock_nowstr):
        with io.BytesIO() as stream:
            joblib.dump(int, stream)

            pickle_bytes = stream.getvalue()
            pickle_b64_str = bytes_to_b64_str(pickle_bytes)

        item1 = PickleItem.factory(int)
        item2 = PickleItem(
            pickle_b64_str=pickle_b64_str,
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        assert item1.object is int
        assert item2.object is int
