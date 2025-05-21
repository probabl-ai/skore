from json import dumps

import numpy as np
from polars import DataFrame
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal
from pytest import raises
from skore_hub_project.item import PolarsDataFrameItem
from skore_hub_project.item.item import ItemTypeError


class TestPolarsDataFrameItem:
    def test_factory(self):
        dataframe = DataFrame([{"key": "value"}])
        dataframe_json_str = dataframe.write_json()

        item = PolarsDataFrameItem.factory(dataframe)

        assert item.dataframe_json_str == dataframe_json_str

    def test_factory_exception(self):
        with raises(ComputeError):
            PolarsDataFrameItem.factory(DataFrame([{"key": np.array([1, 2])}]))

        with raises(ItemTypeError):
            PolarsDataFrameItem.factory(None)

    def test_raw(self):
        dataframe = DataFrame([{"key": "value"}])
        dataframe_json_str = dataframe.write_json()

        item1 = PolarsDataFrameItem.factory(dataframe)
        item2 = PolarsDataFrameItem(dataframe_json_str)

        assert_frame_equal(item1.__raw__, dataframe)
        assert_frame_equal(item2.__raw__, dataframe)

    def test_parameters(self):
        dataframe = DataFrame([{"key": "value"}])
        dataframe_json_str = dataframe.write_json()

        item = PolarsDataFrameItem.factory(dataframe)
        item_parameters = item.__parameters__

        assert item_parameters == {
            "parameters": {
                "class": "PolarsDataFrameItem",
                "parameters": {"dataframe_json_str": dataframe_json_str},
            }
        }

        # Ensure parameters are JSONable
        dumps(item_parameters)

    def test_representation(self):
        dataframe = DataFrame([{"key": "value"}])
        dataframe_json_str = dataframe.write_json()
        representation = {
            "representation": {
                "media_type": "application/vnd.dataframe",
                "value": {
                    "index": [0],
                    "columns": ["key"],
                    "data": [["value"]],
                    "index_names": [None],
                    "column_names": [None],
                },
            }
        }

        item1 = PolarsDataFrameItem.factory(dataframe)
        item2 = PolarsDataFrameItem(dataframe_json_str)

        assert item1.__representation__ == representation
        assert item2.__representation__ == representation

        # Ensure representation is JSONable
        dumps(item1.__representation__)
        dumps(item2.__representation__)
