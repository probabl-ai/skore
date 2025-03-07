import numpy as np
import pytest
from pandas import Index, MultiIndex, Series
from pandas.testing import assert_series_equal
from skore.persistence.item import ItemTypeError, PandasSeriesItem


class TestPandasSeriesItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            PandasSeriesItem.factory(None)

    @pytest.mark.order(0)
    def test_factory(self, mock_nowstr):
        series = Series([0, 1, 2], Index([0, 1, 2], name="myIndex"))

        orient = PandasSeriesItem.ORIENT
        index_json = series.index.to_frame(index=False).to_json(orient=orient)
        series_json = series.reset_index(drop=True).to_json(orient=orient)

        item = PandasSeriesItem.factory(series)

        assert item.index_json == index_json
        assert item.series_json == series_json
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    @pytest.mark.order(1)
    def test_series(self, mock_nowstr):
        series = Series([0, 1, 2], Index([0, 1, 2], name="myIndex"))

        orient = PandasSeriesItem.ORIENT
        index_json = series.index.to_frame(index=False).to_json(orient=orient)
        series_json = series.reset_index(drop=True).to_json(orient=orient)

        item1 = PandasSeriesItem.factory(series)
        item2 = PandasSeriesItem(
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
        item = PandasSeriesItem.factory(series)

        # NOTE: isinstance would not work because numpy.array is an instance of list
        assert type(item.series.iloc[0]) is list

    @pytest.mark.order(1)
    def test_series_with_integer_indexes_name_and_multi_index(self, mock_nowstr):
        series = Series(
            [">70", ">70"],
            MultiIndex.from_arrays(
                [["france", "usa"], ["paris", "nyc"], ["1", "1"]],
                names=(0, "city", "district"),
            ),
        )

        orient = PandasSeriesItem.ORIENT
        index_json = series.index.to_frame(index=False).to_json(orient=orient)
        series_json = series.reset_index(drop=True).to_json(orient=orient)

        item1 = PandasSeriesItem.factory(series)
        item2 = PandasSeriesItem(
            index_json=index_json,
            series_json=series_json,
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        assert_series_equal(item1.series, series)
        assert_series_equal(item2.series, series)
