import json

from altair import Chart
from pytest import fixture, raises
from skore.hub.item import AltairChartItem
from skore.hub.item.item import ItemTypeError, Representation


class TestAltairChartItem:
    @fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.hub.item.item.datetime", MockDatetime)

    def test_factory(self, mock_nowstr):
        chart = Chart().mark_point()
        item = AltairChartItem.factory(chart)

        assert item.chart_json_str == chart.to_json()
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr
        assert item.note == None

    def test_factory_exception(self):
        with raises(ItemTypeError):
            AltairChartItem.factory(None)

    def test_parameters(self, mock_nowstr):
        chart = Chart().mark_point()
        item = AltairChartItem.factory(chart)
        item_parameters = item.__parameters__

        assert item_parameters == {
            "chart_json_str": chart.to_json(),
            "created_at": mock_nowstr,
            "updated_at": mock_nowstr,
            "note": None,
        }

        # Ensure parameters are JSONable
        json.dumps(item_parameters)

    def test_raw(self):
        chart = Chart().mark_point()
        chart_json_str = chart.to_json()
        item1 = AltairChartItem.factory(chart)
        item2 = AltairChartItem(chart_json_str)

        # Altair strict equality doesn't work
        assert item1.__raw__.to_json() == chart_json_str
        assert item2.__raw__.to_json() == chart_json_str

    def test_representation(self):
        chart = Chart().mark_point()
        chart_json_str = chart.to_json()
        representation = Representation(
            media_type="application/vnd.vega.v5+json",
            value=chart_json_str,
        )

        item1 = AltairChartItem.factory(chart)
        item2 = AltairChartItem(chart_json_str)

        assert item1.__representation__ == representation
        assert item2.__representation__ == representation
