import base64
import json

import plotly.graph_objects
import plotly.io
import pytest
from skore.persistence.item import ItemTypeError, PlotlyFigureItem


class TestPlotlyFigureItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)

    def test_factory(self, mock_nowstr):
        bar = plotly.graph_objects.Bar(x=[1, 2, 3], y=[1, 3, 2])
        figure = plotly.graph_objects.Figure(data=[bar])
        item = PlotlyFigureItem.factory(figure)

        assert item.figure_str == plotly.io.to_json(figure, engine="json")
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            PlotlyFigureItem.factory(None)

    def test_ensure_jsonable(self):
        bar = plotly.graph_objects.Bar(x=[1, 2, 3], y=[1, 3, 2])
        figure = plotly.graph_objects.Figure(data=[bar])

        item = PlotlyFigureItem.factory(figure)
        item_parameters = item.__parameters__

        json.dumps(item_parameters)

    def test_figure(self):
        bar = plotly.graph_objects.Bar(x=[1, 2, 3], y=[1, 3, 2])
        figure = plotly.graph_objects.Figure(data=[bar])
        item1 = PlotlyFigureItem.factory(figure)
        item2 = PlotlyFigureItem(plotly.io.to_json(figure, engine="json"))

        assert item1.figure == figure
        assert item2.figure == figure

    def test_as_serializable_dict(self, mock_nowstr):
        bar = plotly.graph_objects.Bar(x=[1, 2, 3], y=[1, 3, 2])
        figure = plotly.graph_objects.Figure(data=[bar])
        figure_str = plotly.io.to_json(figure, engine="json")
        figure_bytes = figure_str.encode("utf-8")
        figure_b64_str = base64.b64encode(figure_bytes).decode()

        item = PlotlyFigureItem.factory(figure)

        assert item.as_serializable_dict() == {
            "updated_at": mock_nowstr,
            "created_at": mock_nowstr,
            "note": None,
            "media_type": "application/vnd.plotly.v1+json;base64",
            "value": figure_b64_str,
        }
