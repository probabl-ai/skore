from json import dumps

import numpy as np
from polars import Series
from polars.testing import assert_series_equal
from pytest import raises
from skore_remote_project.item import PolarsSeriesItem
from skore_remote_project.item.item import ItemTypeError


class TestPolarsSeriesItem:
    def test_factory(self):
        series = Series([0, 1, 2])
        series_json_str = series.to_frame().write_json()

        item = PolarsSeriesItem.factory(series)

        assert item.series_json_str == series_json_str

    def test_factory_exception(self):
        with raises(ItemTypeError):
            PolarsSeriesItem.factory(None)

    def test_parameters(self):
        series = Series([0, 1, 2])
        series_json_str = series.to_frame().write_json()

        item = PolarsSeriesItem.factory(series)
        item_parameters = item.__parameters__

        assert item_parameters == {
            "parameters": {
                "class": "PolarsSeriesItem",
                "parameters": {"series_json_str": series_json_str},
            }
        }

        # Ensure parameters are JSONable
        dumps(item_parameters)

    def test_raw(self):
        series = Series([np.array([1, 2])])
        series_json_str = series.to_frame().write_json()

        item1 = PolarsSeriesItem.factory(series)
        item2 = PolarsSeriesItem(series_json_str)

        assert_series_equal(item1.__raw__, series, check_dtypes=False)
        assert_series_equal(item2.__raw__, series, check_dtypes=False)

    def test_representation(self):
        series = Series([0, 1, 2])
        series_json_str = series.to_frame().write_json()
        representation = {
            "representation": {
                "media_type": "application/json",
                "value": [0, 1, 2],
            }
        }

        item1 = PolarsSeriesItem.factory(series)
        item2 = PolarsSeriesItem(series_json_str)

        assert item1.__representation__ == representation
        assert item2.__representation__ == representation

        # Ensure representation is JSONable
        dumps(item1.__representation__)
        dumps(item2.__representation__)
