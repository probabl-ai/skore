import base64
import io

import altair
import numpy
import pandas
import PIL
import plotly
import polars
import pytest
import sklearn
from matplotlib.figure import Figure
from skore.persistence.item.altair_chart_item import AltairChartItem
from skore.persistence.item.matplotlib_figure_item import MatplotlibFigureItem
from skore.persistence.item.media_item import MediaItem, MediaType
from skore.persistence.item.numpy_array_item import NumpyArrayItem
from skore.persistence.item.pandas_dataframe_item import PandasDataFrameItem
from skore.persistence.item.pandas_series_item import PandasSeriesItem
from skore.persistence.item.pickle_item import PickleItem
from skore.persistence.item.pillow_image_item import PillowImageItem
from skore.persistence.item.plotly_figure_item import PlotlyFigureItem
from skore.persistence.item.polars_dataframe_item import PolarsDataFrameItem
from skore.persistence.item.polars_series_item import PolarsSeriesItem
from skore.persistence.item.primitive_item import PrimitiveItem
from skore.persistence.item.sklearn_base_estimator_item import SklearnBaseEstimatorItem
from skore.ui.serializers import (
    _altair_chart_item_as_serializable_dict,
    _matplotlib_figure_item_as_serializable_dict,
    _media_item_as_serializable_dict,
    _numpy_array_item_as_serializable_dict,
    _pandas_data_frame_item_as_serializable_dict,
    _pandas_series_item_as_serializable_dict,
    _pickle_item_as_serializable_dict,
    _pillow_image_item_as_serializable_dict,
    _plotly_figure_item_as_serializable_dict,
    _polars_data_frame_item_as_serializable_dict,
    _polars_series_item_as_serializable,
    _primitive_item_as_serializable,
    _sklearn_base_estimator_item_as_serializable,
    item_as_serializable,
)
from skore.utils import bytes_to_b64_str


def test_item_as_serializable_dict(monkeypatch, mock_nowstr, MockDatetime):
    monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)

    item = PrimitiveItem.factory(2)
    assert item_as_serializable(item) == {
        "updated_at": mock_nowstr,
        "created_at": mock_nowstr,
        "note": None,
        "media_type": "text/markdown",
        "value": 2,
    }


def test_altair_chart_item_as_serializable_dict():
    chart = altair.Chart().mark_point()
    chart_str = chart.to_json()
    chart_bytes = chart_str.encode("utf-8")
    chart_b64_str = base64.b64encode(chart_bytes).decode()

    item = AltairChartItem.factory(chart)

    assert _altair_chart_item_as_serializable_dict(item) == {
        "media_type": "application/vnd.vega.v5+json;base64",
        "value": chart_b64_str,
    }


class FakeFigure(Figure):
    def savefig(self, stream, *args, **kwargs):
        stream.write(b"<figure>")


def test_matplotlib_figure_item_as_serializable_dict():
    figure = FakeFigure()

    with io.BytesIO() as stream:
        figure.savefig(stream, format="svg", bbox_inches="tight")

        figure_bytes = stream.getvalue()
        figure_b64_str = base64.b64encode(figure_bytes).decode()

    item = MatplotlibFigureItem.factory(figure)

    assert _matplotlib_figure_item_as_serializable_dict(item) == {
        "media_type": "image/svg+xml;base64",
        "value": figure_b64_str,
    }


@pytest.mark.parametrize("media_type", [enum.value for enum in MediaType])
def test_media_item_as_serializable_dict(media_type):
    item = MediaItem.factory("<content>", media_type)

    assert _media_item_as_serializable_dict(item) == {
        "media_type": media_type,
        "value": "<content>",
    }


def test_numpy_array_item_as_serializable_dict():
    array = numpy.array([1, 2, 3])

    item = NumpyArrayItem.factory(array)
    serializable = _numpy_array_item_as_serializable_dict(item)
    assert serializable == {
        "media_type": "text/markdown",
        "value": array.tolist(),
    }


def test_pandas_data_frame_item_as_serializable_dict():
    dataframe = pandas.DataFrame(
        [{"key": numpy.array([1])}], pandas.Index([0], name="myIndex")
    )
    item = PandasDataFrameItem.factory(dataframe)
    serializable = _pandas_data_frame_item_as_serializable_dict(item)
    assert serializable == {
        "media_type": "application/vnd.dataframe",
        "value": dataframe.fillna("NaN").to_dict(orient="tight"),
    }


def test_pandas_series_item_as_serializable_dict():
    series = pandas.Series([numpy.array([1])], pandas.Index([0], name="myIndex"))
    item = PandasSeriesItem.factory(series)
    serializable = _pandas_series_item_as_serializable_dict(item)
    assert serializable == {
        "media_type": "text/markdown",
        "value": series.to_list(),
    }


def test_pickle_item_as_serializable_dict():
    item = PickleItem.factory(int)
    serializable = _pickle_item_as_serializable_dict(item)
    assert serializable == {
        "media_type": "text/markdown",
        "value": "```python\n<class 'int'>\n```",
    }


def test_pillow_image_item_as_serializable_dict():
    image = PIL.Image.new("RGB", (100, 100), color="red")
    item = PillowImageItem.factory(image)

    with io.BytesIO() as stream:
        image.save(stream, format="png")

        png_bytes = stream.getvalue()
        png_b64_str = bytes_to_b64_str(png_bytes)

    assert _pillow_image_item_as_serializable_dict(item) == {
        "media_type": "image/png;base64",
        "value": png_b64_str,
    }


def test_plotly_figure_item_as_serializable_dict():
    bar = plotly.graph_objects.Bar(x=[1, 2, 3], y=[1, 3, 2])
    figure = plotly.graph_objects.Figure(data=[bar])
    figure_str = plotly.io.to_json(figure, engine="json")
    figure_bytes = figure_str.encode("utf-8")
    figure_b64_str = base64.b64encode(figure_bytes).decode()

    item = PlotlyFigureItem.factory(figure)

    assert _plotly_figure_item_as_serializable_dict(item) == {
        "media_type": "application/vnd.plotly.v1+json;base64",
        "value": figure_b64_str,
    }


def test_polars_data_frame_item_as_serializable_dict():
    dataframe = polars.DataFrame([{"key": "value"}])
    item = PolarsDataFrameItem.factory(dataframe)
    serializable = _polars_data_frame_item_as_serializable_dict(item)
    assert serializable == {
        "media_type": "application/vnd.dataframe",
        "value": dataframe.to_pandas().fillna("NaN").to_dict(orient="tight"),
    }


def test_polars_series_item_as_serializable():
    series = polars.Series([numpy.array([1, 2])])
    item = PolarsSeriesItem.factory(series)
    serializable = _polars_series_item_as_serializable(item)
    assert serializable == {
        "media_type": "text/markdown",
        "value": series.to_list(),
    }


@pytest.mark.parametrize(
    "primitive",
    [
        0,
        1.1,
        True,
        [0, 1, 2],
        (0, 1, 2),
        {"a": 0},
    ],
)
def test_primitive_item_as_serializable(monkeypatch, MockDatetime, primitive):
    monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)

    item = PrimitiveItem.factory(primitive)
    serializable = _primitive_item_as_serializable(item)
    assert serializable == {
        "media_type": "text/markdown",
        "value": primitive,
    }


def test_sklearn_base_estimator_item_as_serializable():
    class Estimator(sklearn.svm.SVC):
        pass

    estimator = Estimator()
    item = SklearnBaseEstimatorItem.factory(estimator)
    serializable = _sklearn_base_estimator_item_as_serializable(item)

    assert serializable == {
        "media_type": "application/vnd.sklearn.estimator+html",
        "value": item.estimator_html_repr,
    }
