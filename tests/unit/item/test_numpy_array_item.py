import numpy
import pytest
from skore.item import NumpyArrayItem


class TestNumpyArrayItem:
    @pytest.mark.order(0)
    def test_factory(self):
        array = numpy.array([1, 2, 3])
        array_list = array.tolist()

        item = NumpyArrayItem.factory(array)

        assert vars(item) == {"array_list": array_list}

    @pytest.mark.order(1)
    def test_array(self):
        array = numpy.array([1, 2, 3])
        array_list = array.tolist()

        item1 = NumpyArrayItem.factory(array)
        item2 = NumpyArrayItem(array_list)

        numpy.testing.assert_array_equal(item1.array, array)
        numpy.testing.assert_array_equal(item2.array, array)
