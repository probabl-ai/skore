import json
from io import BytesIO

import numpy
import pytest
from skore.persistence.item import ItemTypeError, NumpyArrayItem
from skore.utils import bytes_to_b64_str


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
        with BytesIO() as stream:
            numpy.save(stream, array)

            array_bytes = stream.getvalue()
            array_b64_str = bytes_to_b64_str(array_bytes)

        item = NumpyArrayItem.factory(array)

        assert item.array_b64_str == array_b64_str
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    @pytest.mark.order(1)
    def test_array(self, mock_nowstr):
        array = numpy.array([1, 2, 3])
        with BytesIO() as stream:
            numpy.save(stream, array)

            array_bytes = stream.getvalue()
            array_b64_str = bytes_to_b64_str(array_bytes)

        item1 = NumpyArrayItem.factory(array)
        item2 = NumpyArrayItem(
            array_b64_str=array_b64_str,
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        numpy.testing.assert_array_equal(item1.array, array)
        numpy.testing.assert_array_equal(item2.array, array)

    def test_ensure_jsonable(self):
        item = NumpyArrayItem.factory(numpy.array([1, 2, 3]))
        item_parameters = item.__parameters__

        json.dumps(item_parameters)

    @pytest.mark.order(1)
    def test_array_with_complex_object(self, mock_nowstr):
        array = numpy.array([object])

        with pytest.raises(
            ValueError,
            match="Object arrays cannot be saved when allow_pickle=False",
        ):
            NumpyArrayItem.factory(array)
