import pytest
from pandas import Series
from pandas.testing import assert_series_equal
from skore.item import ItemTypeError, PandasSeriesItem


class TestPandasSeriesItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.item.item.datetime", MockDatetime)

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            PandasSeriesItem.factory(None)

    @pytest.mark.order(0)
    def test_factory(self, mock_nowstr):
        series = Series([0, 1, 2])
        series_list = series.to_list()

        item = PandasSeriesItem.factory(series)

        assert item.series_list == series_list
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    @pytest.mark.order(1)
    def test_series(self, mock_nowstr):
        series = Series([0, 1, 2])
        series_list = series.to_list()

        item1 = PandasSeriesItem.factory(series)
        item2 = PandasSeriesItem(
            series_list=series_list,
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        assert_series_equal(item1.series, series)
        assert_series_equal(item2.series, series)
