from json import dumps

import numpy as np
from pandas import DataFrame, Index, MultiIndex
from pandas.testing import assert_frame_equal
from pytest import raises
from skore_hub_project.item import PandasDataFrameItem
from skore_hub_project.item.item import ItemTypeError

ORIENT = PandasDataFrameItem.ORIENT


class TestPandasDataFrameItem:
    def test_factory(self):
        dataframe = DataFrame([{"key": "value"}], Index([0], name="myIndex"))

        index_json_str = dataframe.index.to_frame(index=False).to_json(orient=ORIENT)
        dataframe_json_str = dataframe.reset_index(drop=True).to_json(orient=ORIENT)

        item = PandasDataFrameItem.factory(dataframe)

        assert item.index_json_str == index_json_str
        assert item.dataframe_json_str == dataframe_json_str

    def test_factory_exception(self):
        with raises(ItemTypeError):
            PandasDataFrameItem.factory(None)

    def test_raw(self):
        dataframe = DataFrame([{"key": "value"}], Index([0], name="myIndex"))

        index_json_str = dataframe.index.to_frame(index=False).to_json(orient=ORIENT)
        dataframe_json_str = dataframe.reset_index(drop=True).to_json(orient=ORIENT)

        item1 = PandasDataFrameItem.factory(dataframe)
        item2 = PandasDataFrameItem(index_json_str, dataframe_json_str)

        assert_frame_equal(item1.__raw__, dataframe)
        assert_frame_equal(item2.__raw__, dataframe)

    def test_raw_with_complex_object(self):
        dataframe = DataFrame([{"key": np.array([1])}], Index([0], name="myIndex"))

        index_json_str = dataframe.index.to_frame(index=False).to_json(orient=ORIENT)
        dataframe_json_str = dataframe.reset_index(drop=True).to_json(orient=ORIENT)

        item1 = PandasDataFrameItem.factory(dataframe)
        item2 = PandasDataFrameItem(index_json_str, dataframe_json_str)

        assert type(item1.__raw__["key"].iloc[0]) is np.ndarray
        assert type(item2.__raw__["key"].iloc[0]) is list

    def test_raw_with_integer_columns_name_and_multiindex(self):
        dataframe = DataFrame(
            [[">70", "1M", "M", 1], [">70", "2F", "F", 2]],
            MultiIndex.from_arrays(
                [["france", "usa"], ["paris", "nyc"], ["1", "1"]],
                names=("country", "city", "district"),
            ),
        )

        index_json_str = dataframe.index.to_frame(index=False).to_json(orient=ORIENT)
        dataframe_json_str = dataframe.reset_index(drop=True).to_json(orient=ORIENT)

        item1 = PandasDataFrameItem.factory(dataframe)
        item2 = PandasDataFrameItem(index_json_str, dataframe_json_str)

        assert_frame_equal(item1.__raw__, dataframe)
        assert_frame_equal(item2.__raw__, dataframe)

    def test_parameters(self):
        dataframe = DataFrame([{"key": "value"}], Index([0], name="myIndex"))

        index_json_str = dataframe.index.to_frame(index=False).to_json(orient=ORIENT)
        dataframe_json_str = dataframe.reset_index(drop=True).to_json(orient=ORIENT)

        item = PandasDataFrameItem.factory(dataframe)
        item_parameters = item.__parameters__

        assert item_parameters == {
            "parameters": {
                "class": "PandasDataFrameItem",
                "parameters": {
                    "index_json_str": index_json_str,
                    "dataframe_json_str": dataframe_json_str,
                },
            }
        }

        # Ensure parameters are JSONable
        dumps(item_parameters)

    def test_representation(self):
        dataframe = DataFrame([{"key": "value"}], Index([0], name="myIndex"))
        representation = {
            "representation": {
                "media_type": "application/vnd.dataframe",
                "value": {
                    "index": [0],
                    "columns": ["key"],
                    "data": [["value"]],
                    "index_names": ["myIndex"],
                    "column_names": [None],
                },
            }
        }

        index_json_str = dataframe.index.to_frame(index=False).to_json(orient=ORIENT)
        dataframe_json_str = dataframe.reset_index(drop=True).to_json(orient=ORIENT)

        item1 = PandasDataFrameItem.factory(dataframe)
        item2 = PandasDataFrameItem(index_json_str, dataframe_json_str)

        assert item1.__representation__ == representation
        assert item2.__representation__ == representation

        # Ensure representation is JSONable
        dumps(item1.__representation__)
        dumps(item2.__representation__)
