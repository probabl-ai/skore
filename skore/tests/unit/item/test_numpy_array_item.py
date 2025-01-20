import json

import numpy
import pytest
from skore.persistence.item import ItemTypeError, NumpyArrayItem


class TestNumpyArrayItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            NumpyArrayItem.factory(None)

    @pytest.mark.order(0)
    def test_factory(self, mock_nowstr):
        array = numpy.array([1, 2, 3])
        array_json = json.dumps(array.tolist())

        item = NumpyArrayItem.factory(array)

        assert item.array_json == array_json
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    @pytest.mark.order(1)
    def test_array(self, mock_nowstr):
        array = numpy.array([1, 2, 3])
        array_json = json.dumps(array.tolist())

        item1 = NumpyArrayItem.factory(array)
        item2 = NumpyArrayItem(
            array_json=array_json,
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        numpy.testing.assert_array_equal(item1.array, array)
        numpy.testing.assert_array_equal(item2.array, array)

    @pytest.mark.order(1)
    def test_array_with_complex_object(self, mock_nowstr):
        array = numpy.array([object])

        with pytest.raises(TypeError, match="type is not JSON serializable"):
            NumpyArrayItem.factory(array)

    def test_get_serializable_dict(self, mock_nowstr):
        array = numpy.array([1, 2, 3])

        item = NumpyArrayItem.factory(array)
        serializable = item.as_serializable_dict()
        assert serializable == {
            "updated_at": mock_nowstr,
            "created_at": mock_nowstr,
            "note": None,
            "media_type": "text/markdown",
            "value": array.tolist(),
        }
