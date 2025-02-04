import json

import altair
import pytest
from skore.persistence.item import AltairChartItem, ItemTypeError


class TestAltairChartItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)

    def test_factory(self, mock_nowstr):
        chart = altair.Chart().mark_point()
        item = AltairChartItem.factory(chart)

        assert item.chart_str == chart.to_json()
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            AltairChartItem.factory(None)

    def test_ensure_jsonable(self):
        chart = altair.Chart().mark_point()

        item = AltairChartItem.factory(chart)
        item_parameters = item.__parameters__

        json.dumps(item_parameters)

    def test_chart(self):
        chart = altair.Chart().mark_point()
        chart_str = chart.to_json()
        item1 = AltairChartItem.factory(chart)
        item2 = AltairChartItem(chart_str)

        # Altair strict equality doesn't work
        assert item1.chart.to_json() == chart_str
        assert item2.chart.to_json() == chart_str
