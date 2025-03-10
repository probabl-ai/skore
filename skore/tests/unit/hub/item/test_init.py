import altair
import matplotlib.pyplot as plt
import numpy
import pandas
import PIL.Image
import plotly.graph_objects
import polars
import pytest
import sklearn
import skrub
from skore.hub.item import (
    AltairChartItem,
    JSONableItem,
    MatplotlibFigureItem,
    NumpyArrayItem,
    PandasDataFrameItem,
    PandasSeriesItem,
    PickleItem,
    PillowImageItem,
    PlotlyFigureItem,
    PolarsDataFrameItem,
    PolarsSeriesItem,
    SklearnBaseEstimatorItem,
    SkrubTableReportItem,
    object_to_item,
)


@pytest.mark.parametrize(
    "object, item_cls",
    (
        (altair.Chart().mark_point(), AltairChartItem),
        ("value", JSONableItem),
        (plt.subplots()[0], MatplotlibFigureItem),
        (numpy.array([0, 1, 2]), NumpyArrayItem),
        (pandas.DataFrame([{"key": "value"}]), PandasDataFrameItem),
        (pandas.Series([0, 1, 2]), PandasSeriesItem),
        (int, PickleItem),
        (PIL.Image.new("RGB", (100, 100), color="red"), PillowImageItem),
        (
            plotly.graph_objects.Figure(
                data=[
                    plotly.graph_objects.Bar(
                        x=[1, 2, 3],
                        y=[1, 3, 2],
                    )
                ]
            ),
            PlotlyFigureItem,
        ),
        (polars.DataFrame([{"key": "value"}]), PolarsDataFrameItem),
        (polars.Series([0, 1, 2]), PolarsSeriesItem),
        (sklearn.svm.SVC(), SklearnBaseEstimatorItem),
        (
            skrub.TableReport(
                pandas.DataFrame(
                    {
                        "a": [1, 2],
                        "b": ["one", "two"],
                        "c": [11.1, 11.1],
                    }
                )
            ),
            SkrubTableReportItem,
        ),
    ),
)
def test_object_to_item(object, item_cls):
    assert isinstance(object_to_item(object), item_cls)


def test_object_to_item_media_item(): ...
