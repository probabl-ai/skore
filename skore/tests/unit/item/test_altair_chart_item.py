import base64
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

    def test_as_serializable_dict(self, mock_nowstr):
        chart = altair.Chart().mark_point()
        chart_str = chart.to_json()
        chart_bytes = chart_str.encode("utf-8")
        chart_b64_str = base64.b64encode(chart_bytes).decode()

        item = AltairChartItem.factory(chart)

        assert item.as_serializable_dict() == {
            "updated_at": mock_nowstr,
            "created_at": mock_nowstr,
            "note": None,
            "media_type": "application/vnd.vega.v5+json;base64",
            "value": chart_b64_str,
        }
