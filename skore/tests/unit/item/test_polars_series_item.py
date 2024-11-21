import numpy as np
import pytest
from polars import Index, MultiIndex, Series
from polars.testing import assert_series_equal
from skore.item import ItemTypeError, PolarsSeriesItem


class TestPolarsSeriesItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.item.item.datetime", MockDatetime)

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            PolarsSeriesItem.factory(None)

    @pytest.mark.order(0)
    def test_factory(self, mock_nowstr):
        series = Series([0, 1, 2], Index([0, 1, 2], name="myIndex"))

        orient = PolarsSeriesItem.ORIENT
        index_json = series.index.to_frame(index=False).to_json(orient=orient)
        series_json = series.reset_index(drop=True).to_json(orient=orient)

        item = PolarsSeriesItem.factory(series)

        assert item.index_json == index_json
        assert item.series_json == series_json
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    @pytest.mark.order(1)
    def test_series(self, mock_nowstr):
        series = Series([0, 1, 2], Index([0, 1, 2], name="myIndex"))

        orient = PolarsSeriesItem.ORIENT
        index_json = series.index.to_frame(index=False).to_json(orient=orient)
        series_json = series.reset_index(drop=True).to_json(orient=orient)

        item1 = PolarsSeriesItem.factory(series)
        item2 = PolarsSeriesItem(
            index_json=index_json,
            series_json=series_json,
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        assert_series_equal(item1.series, series)
        assert_series_equal(item2.series, series)

    @pytest.mark.order(1)
    def test_series_with_complex_object(self, mock_nowstr):
        series = Series([np.array([1])], Index([0], name="myIndex"))
        item = PolarsSeriesItem.factory(series)

        assert type(item.series.iloc[0]) is list

    @pytest.mark.order(1)
    def test_series_with_integer_indexes_name_and_multiindex(self, mock_nowstr):
        series = Series(
            [">70", ">70"],
            MultiIndex.from_arrays(
                [["france", "usa"], ["paris", "nyc"], ["1", "1"]],
                names=(0, "city", "district"),
            ),
        )

        orient = PolarsSeriesItem.ORIENT
        index_json = series.index.to_frame(index=False).to_json(orient=orient)
        series_json = series.reset_index(drop=True).to_json(orient=orient)

        item1 = PolarsSeriesItem.factory(series)
        item2 = PolarsSeriesItem(
            index_json=index_json,
            series_json=series_json,
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        assert_series_equal(item1.series, series)
        assert_series_equal(item2.series, series)
