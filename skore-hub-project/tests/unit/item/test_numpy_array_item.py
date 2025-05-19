import json
from io import BytesIO

import numpy
from pytest import raises
from skore_remote_project.item import NumpyArrayItem
from skore_remote_project.item.item import ItemTypeError, bytes_to_b64_str


class TestNumpyArrayItem:
    def test_factory(self):
        array = numpy.array([1, 2, 3])

        with BytesIO() as stream:
            numpy.save(stream, array)

            array_bytes = stream.getvalue()
            array_b64_str = bytes_to_b64_str(array_bytes)

        item = NumpyArrayItem.factory(array)

        assert item.array_b64_str == array_b64_str

    def test_factory_exception(self):
        with raises(ValueError, match="Object arrays cannot be saved"):
            NumpyArrayItem.factory(numpy.array([object]))

        with raises(ItemTypeError):
            NumpyArrayItem.factory(None)

    def test_raw(self):
        array = numpy.array([1, 2, 3])

        with BytesIO() as stream:
            numpy.save(stream, array)

            array_bytes = stream.getvalue()
            array_b64_str = bytes_to_b64_str(array_bytes)

        item1 = NumpyArrayItem.factory(array)
        item2 = NumpyArrayItem(array_b64_str)

        numpy.testing.assert_array_equal(item1.__raw__, array)
        numpy.testing.assert_array_equal(item2.__raw__, array)

    def test_parameters(self):
        array = numpy.array([1, 2, 3])

        with BytesIO() as stream:
            numpy.save(stream, array)

            array_bytes = stream.getvalue()
            array_b64_str = bytes_to_b64_str(array_bytes)

        item = NumpyArrayItem.factory(array)
        item_parameters = item.__parameters__

        assert item_parameters == {
            "parameters": {
                "class": "NumpyArrayItem",
                "parameters": {"array_b64_str": array_b64_str},
            }
        }

        # Ensure parameters are JSONable
        json.dumps(item_parameters)

    def test_representation(self):
        array = numpy.array([1, 2, 3])
        representation = {
            "representation": {
                "media_type": "application/json",
                "value": [1, 2, 3],
            }
        }

        with BytesIO() as stream:
            numpy.save(stream, array)

            array_bytes = stream.getvalue()
            array_b64_str = bytes_to_b64_str(array_bytes)

        item1 = NumpyArrayItem.factory(array)
        item2 = NumpyArrayItem(array_b64_str)

        assert item1.__representation__ == representation
        assert item2.__representation__ == representation

        # Ensure representation is JSONable
        json.dumps(item1.__representation__)
        json.dumps(item2.__representation__)
