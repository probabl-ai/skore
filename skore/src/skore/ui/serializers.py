"""Centralize all item serialization functions."""

from io import BytesIO
from traceback import format_exception

from skore.persistence.item import (
    AltairChartItem,
    Item,
    MatplotlibFigureItem,
    MediaItem,
    NumpyArrayItem,
    PandasDataFrameItem,
    PandasSeriesItem,
    PickleItem,
    PillowImageItem,
    PlotlyFigureItem,
    PolarsDataFrameItem,
    PolarsSeriesItem,
    PrimitiveItem,
    SklearnBaseEstimatorItem,
)
from skore.utils import bytes_to_b64_str


def item_as_serializable(item: Item):
    """Convert item to a JSON-serializable dict."""
    subclass = {}
    if isinstance(item, PrimitiveItem):
        subclass = _primitive_item_as_serializable(item)
    elif isinstance(item, NumpyArrayItem):
        subclass = _numpy_array_item_as_serializable_dict(item)
    elif isinstance(item, PandasDataFrameItem):
        subclass = _pandas_data_frame_item_as_serializable_dict(item)
    elif isinstance(item, PolarsDataFrameItem):
        subclass = _polars_data_frame_item_as_serializable_dict(item)
    elif isinstance(item, PandasSeriesItem):
        subclass = _pandas_series_item_as_serializable_dict(item)
    elif isinstance(item, PolarsSeriesItem):
        subclass = _polars_series_item_as_serializable(item)
    elif isinstance(item, SklearnBaseEstimatorItem):
        subclass = _sklearn_base_estimator_item_as_serializable(item)
    elif isinstance(item, MediaItem):
        subclass = _media_item_as_serializable_dict(item)
    elif isinstance(item, PillowImageItem):
        subclass = _pillow_image_item_as_serializable_dict(item)
    elif isinstance(item, PlotlyFigureItem):
        subclass = _plotly_figure_item_as_serializable_dict(item)
    elif isinstance(item, AltairChartItem):
        subclass = _altair_chart_item_as_serializable_dict(item)
    elif isinstance(item, MatplotlibFigureItem):
        subclass = _matplotlib_figure_item_as_serializable_dict(item)
    elif isinstance(item, PickleItem):
        subclass = _pickle_item_as_serializable_dict(item)
    return {
        "updated_at": item.updated_at,
        "created_at": item.created_at,
        "note": item.note,
    } | subclass


def _altair_chart_item_as_serializable_dict(item: AltairChartItem):
    """Convert item to a JSON-serializable dict to used by frontend."""
    chart_bytes = item.chart_str.encode("utf-8")
    chart_b64_str = bytes_to_b64_str(chart_bytes)

    return {
        "media_type": "application/vnd.vega.v5+json;base64",
        "value": chart_b64_str,
    }


def _matplotlib_figure_item_as_serializable_dict(item: MatplotlibFigureItem) -> dict:
    """Convert a MatplotlibFigureItem to a JSON-serializable dict."""
    with BytesIO() as stream:
        item.figure.savefig(stream, format="svg", bbox_inches="tight")

        figure_bytes = stream.getvalue()
        figure_b64_str = bytes_to_b64_str(figure_bytes)

        return {
            "media_type": "image/svg+xml;base64",
            "value": figure_b64_str,
        }


def _media_item_as_serializable_dict(item: MediaItem):
    """Convert a MediaItem to a JSON-serializable dict."""
    return {
        "media_type": item.media_type,
        "value": item.media,
    }


def _numpy_array_item_as_serializable_dict(item: NumpyArrayItem):
    """Convert a NumpyArrayItem to a JSON-serializable dict."""
    return {
        "media_type": "text/markdown",
        "value": item.array.tolist(),
    }


def _pandas_data_frame_item_as_serializable_dict(item: PandasDataFrameItem):
    """Convert a PandasDataFrameItem to a JSON-serializable dict."""
    return {
        "media_type": "application/vnd.dataframe",
        "value": item.dataframe.fillna("NaN").to_dict(orient="tight"),
    }


def _pandas_series_item_as_serializable_dict(item: PandasSeriesItem):
    """Convert a PandasSeriesItem to a JSON-serializable dict."""
    return {
        "value": item.series.fillna("NaN").to_list(),
        "media_type": "text/markdown",
    }


def _pickle_item_as_serializable_dict(item: PickleItem):
    """Convert a PickleItem to a JSON-serializable dict."""
    try:
        object = item.object
    except Exception as e:
        traceback = "".join(format_exception(None, e, e.__traceback__))
        value = "".join(
            (
                "Item cannot be displayed",
                "\n\n",
                "UnpicklingError with complete traceback:",
                "\n\n```pytb",
                traceback,
                "```",
            )
        )
    else:
        value = f"```python\n{repr(object)}\n```"

    return {
        "media_type": "text/markdown",
        "value": value,
    }


def _pillow_image_item_as_serializable_dict(item: PillowImageItem):
    """Convert a PillowImageItem to a JSON-serializable dict."""
    with BytesIO() as stream:
        item.image.save(stream, format="png")

        png_bytes = stream.getvalue()
        png_b64_str = bytes_to_b64_str(png_bytes)

    return {
        "media_type": "image/png;base64",
        "value": png_b64_str,
    }


def _plotly_figure_item_as_serializable_dict(item: PlotlyFigureItem):
    """Convert a PlotlyFigureItem to a JSON-serializable dict."""
    figure_bytes = item.figure_str.encode("utf-8")
    figure_b64_str = bytes_to_b64_str(figure_bytes)

    return {
        "media_type": "application/vnd.plotly.v1+json;base64",
        "value": figure_b64_str,
    }


def _polars_data_frame_item_as_serializable_dict(item: PolarsDataFrameItem):
    """Convert a PolarsDataFrameItem to a JSON-serializable dict."""
    return {
        "value": item.dataframe.to_pandas().fillna("NaN").to_dict(orient="tight"),
        "media_type": "application/vnd.dataframe",
    }


def _polars_series_item_as_serializable(item: PolarsSeriesItem):
    """Convert a PolarsSeriesItem to a JSON-serializable dict."""
    return {
        "value": item.series.to_list(),
        "media_type": "text/markdown",
    }


def _primitive_item_as_serializable(item: PrimitiveItem):
    """Convert a PrimitiveItem to a JSON-serializable dict."""
    return {
        "media_type": "text/markdown",
        "value": item.primitive,
    }


def _sklearn_base_estimator_item_as_serializable(item: SklearnBaseEstimatorItem):
    """Convert a SklearnBaseEstimatorItem to a JSON-serializable dict."""
    return {
        "value": item.estimator_html_repr,
        "media_type": "application/vnd.sklearn.estimator+html",
    }
