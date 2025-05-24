from json import dumps

import numpy as np
from pandas import Index, MultiIndex, Series
from pandas.testing import assert_series_equal
from pytest import raises
from skore_hub_project.item import PandasSeriesItem
from skore_hub_project.item.item import ItemTypeError

ORIENT = PandasSeriesItem.ORIENT


class TestPandasSeriesItem:
    def test_factory(self):
        series = Series([0, 1, 2], Index([0, 1, 2], name="myIndex"))

        index_json_str = series.index.to_frame(index=False).to_json(orient=ORIENT)
        series_json_str = series.reset_index(drop=True).to_json(orient=ORIENT)

        item = PandasSeriesItem.factory(series)

        assert item.index_json_str == index_json_str
        assert item.series_json_str == series_json_str

    def test_factory_exception(self):
        with raises(ItemTypeError):
            PandasSeriesItem.factory(None)

    def test_parameters(self):
        series = Series([0, 1, 2], Index([0, 1, 2], name="myIndex"))

        index_json_str = series.index.to_frame(index=False).to_json(orient=ORIENT)
        series_json_str = series.reset_index(drop=True).to_json(orient=ORIENT)

        item = PandasSeriesItem.factory(series)
        item_parameters = item.__parameters__

        assert item_parameters == {
            "parameters": {
                "class": "PandasSeriesItem",
                "parameters": {
                    "index_json_str": index_json_str,
                    "series_json_str": series_json_str,
                },
            }
        }

        # Ensure parameters are JSONable
        dumps(item_parameters)

    def test_raw(self):
        series = Series([0, 1, 2], Index([0, 1, 2], name="myIndex"))

        index_json_str = series.index.to_frame(index=False).to_json(orient=ORIENT)
        series_json_str = series.reset_index(drop=True).to_json(orient=ORIENT)

        item1 = PandasSeriesItem.factory(series)
        item2 = PandasSeriesItem(index_json_str, series_json_str)

        assert_series_equal(item1.__raw__, series)
        assert_series_equal(item2.__raw__, series)

    def test_raw_with_complex_object(self):
        series = Series([np.array([1])], Index([0], name="myIndex"))

        index_json_str = series.index.to_frame(index=False).to_json(orient=ORIENT)
        series_json_str = series.reset_index(drop=True).to_json(orient=ORIENT)

        item1 = PandasSeriesItem.factory(series)
        item2 = PandasSeriesItem(index_json_str, series_json_str)

        assert type(item1.__raw__.iloc[0]) is np.ndarray
        assert type(item2.__raw__.iloc[0]) is list

    def test_raw_with_integer_indexes_name_and_multiindex(self):
        series = Series(
            [">70", ">70"],
            MultiIndex.from_arrays(
                [["france", "usa"], ["paris", "nyc"], ["1", "1"]],
                names=(0, "city", "district"),
            ),
        )

        index_json_str = series.index.to_frame(index=False).to_json(orient=ORIENT)
        series_json_str = series.reset_index(drop=True).to_json(orient=ORIENT)

        item1 = PandasSeriesItem.factory(series)
        item2 = PandasSeriesItem(index_json_str, series_json_str)

        assert_series_equal(item1.__raw__, series)
        assert_series_equal(item2.__raw__, series)

    def test_representation(self):
        series = Series([0, 1, 2], Index([0, 1, 2], name="myIndex"))
        representation = {
            "representation": {
                "media_type": "application/json",
                "value": [0, 1, 2],
            }
        }

        index_json_str = series.index.to_frame(index=False).to_json(orient=ORIENT)
        series_json_str = series.reset_index(drop=True).to_json(orient=ORIENT)

        item1 = PandasSeriesItem.factory(series)
        item2 = PandasSeriesItem(index_json_str, series_json_str)

        assert item1.__representation__ == representation
        assert item2.__representation__ == representation

        # Ensure representation is JSONable
        dumps(item1.__representation__)
        dumps(item2.__representation__)
