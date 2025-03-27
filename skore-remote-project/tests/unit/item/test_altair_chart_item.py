from json import dumps, loads

from altair import Chart
from pytest import raises
from skore_remote_project.item import AltairChartItem
from skore_remote_project.item.item import ItemTypeError


class TestAltairChartItem:
    def test_factory(self):
        chart = Chart().mark_point()
        item = AltairChartItem.factory(chart)

        assert item.chart_json_str == chart.to_json()

    def test_factory_exception(self):
        with raises(ItemTypeError):
            AltairChartItem.factory(None)

    def test_parameters(self):
        chart = Chart().mark_point()
        item = AltairChartItem.factory(chart)
        item_parameters = item.__parameters__

        assert item_parameters == {
            "parameters": {
                "class": "AltairChartItem",
                "parameters": {"chart_json_str": chart.to_json()},
            }
        }

        # Ensure parameters are JSONable
        dumps(item_parameters)

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
        representation = {
            "representation": {
                "media_type": "application/vnd.vega.v5+json",
                "value": loads(chart_json_str),
            }
        }

        item1 = AltairChartItem.factory(chart)
        item2 = AltairChartItem(chart_json_str)

        assert item1.__representation__ == representation
        assert item2.__representation__ == representation

        # Ensure representation is JSONable
        dumps(item1.__representation__)
        dumps(item2.__representation__)
