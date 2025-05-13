from json import dumps, loads

import plotly.graph_objects
import plotly.io
import pytest
from skore_remote_project.item import PlotlyFigureItem
from skore_remote_project.item.item import ItemTypeError


class TestPlotlyFigureItem:
    def test_factory(self):
        bar = plotly.graph_objects.Bar(x=[1, 2, 3], y=[1, 3, 2])
        figure = plotly.graph_objects.Figure(data=[bar])
        figure_json_str = plotly.io.to_json(figure, engine="json")

        item = PlotlyFigureItem.factory(figure)

        assert item.figure_json_str == figure_json_str

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            PlotlyFigureItem.factory(None)

    def test_parameters(self):
        bar = plotly.graph_objects.Bar(x=[1, 2, 3], y=[1, 3, 2])
        figure = plotly.graph_objects.Figure(data=[bar])
        figure_json_str = plotly.io.to_json(figure, engine="json")

        item = PlotlyFigureItem.factory(figure)
        item_parameters = item.__parameters__

        assert item_parameters == {
            "parameters": {
                "class": "PlotlyFigureItem",
                "parameters": {"figure_json_str": figure_json_str},
            }
        }

        # Ensure parameters are JSONable
        dumps(item_parameters)

    def test_raw(self):
        bar = plotly.graph_objects.Bar(x=[1, 2, 3], y=[1, 3, 2])
        figure = plotly.graph_objects.Figure(data=[bar])
        figure_json_str = plotly.io.to_json(figure, engine="json")

        item1 = PlotlyFigureItem.factory(figure)
        item2 = PlotlyFigureItem(figure_json_str)

        assert item1.__raw__ == figure
        assert item2.__raw__ == figure

    def test_representation(self):
        bar = plotly.graph_objects.Bar(x=[1, 2, 3], y=[1, 3, 2])
        figure = plotly.graph_objects.Figure(data=[bar])
        figure_json_str = plotly.io.to_json(figure, engine="json")
        representation = {
            "representation": {
                "media_type": "application/vnd.plotly.v1+json",
                "value": loads(figure_json_str),
            }
        }

        item1 = PlotlyFigureItem.factory(figure)
        item2 = PlotlyFigureItem(figure_json_str)

        assert item1.__representation__ == representation
        assert item2.__representation__ == representation

        # Ensure representation is JSONable
        dumps(item1.__representation__)
        dumps(item2.__representation__)
