import pickle

import pytest
from skore.persistence.item import PickleItem


class TestPickleItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)

    @pytest.mark.parametrize("object", [0, 0.0, int, True, [0], {0: 0}])
    def test_factory(self, mock_nowstr, object):
        item = PickleItem.factory(object)

        assert item.pickle_bytes == pickle.dumps(object)
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    def test_object(self, mock_nowstr):
        item1 = PickleItem.factory(int)
        item2 = PickleItem(
            pickle_bytes=pickle.dumps(int),
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        assert item1.object is int
        assert item2.object is int

    def test_get_serializable_dict(self, mock_nowstr):
        item = PickleItem.factory(int)
        serializable = item.as_serializable_dict()

        assert serializable == {
            "updated_at": mock_nowstr,
            "created_at": mock_nowstr,
            "media_type": "text/markdown",
            "value": repr(int),
        }
