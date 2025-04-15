import numpy as np
import pytest
from polars import Series
from polars.testing import assert_series_equal
from skore.persistence.item import ItemTypeError, PolarsSeriesItem


class TestPolarsSeriesItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            PolarsSeriesItem.factory(None)

    @pytest.mark.order(0)
    def test_factory(self, mock_nowstr):
        series = Series([0, 1, 2])

        series_json = series.to_frame().write_json()

        item = PolarsSeriesItem.factory(series)

        assert item.series_json == series_json
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    @pytest.mark.order(1)
    def test_series(self, mock_nowstr):
        series = Series([0, 1, 2])

        series_json = series.to_frame().write_json()

        item1 = PolarsSeriesItem.factory(series)
        item2 = PolarsSeriesItem(
            series_json=series_json,
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        assert_series_equal(item1.series, series)
        assert_series_equal(item2.series, series)

    @pytest.mark.order(1)
    def test_series_with_complex_object(self, mock_nowstr):
        series = Series([np.array([1, 2])])
        item = PolarsSeriesItem.factory(series)

        assert_series_equal(item.series, series, check_dtypes=False)
