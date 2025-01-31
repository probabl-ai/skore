import base64
import datetime
import io

import altair
import matplotlib.figure
import numpy
import pandas
import plotly
import polars
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sklearn.linear_model import Lasso
from skore.ui.app import create_app


@pytest.fixture
def client(in_memory_project):
    return TestClient(app=create_app(project=in_memory_project))


@pytest.fixture
def monkeypatch_datetime(monkeypatch, MockDatetime):
    monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)


def test_app_state(client):
    assert client.app.state.project is not None


def test_serialize_pandas_dataframe_with_missing_values(client, in_memory_project):
    pandas_df = pandas.DataFrame([1, 2, 3, 4, None, float("nan")])
    in_memory_project.put("🐼", pandas_df)
    response = client.get("/api/project/activity")
    assert response.status_code == 200
    feed = response.json()
    assert len(feed[0]["value"]["data"]) == 6


def test_serialize_polars_dataframe_with_missing_values(client, in_memory_project):
    polars_df = polars.DataFrame([1, 2, 3, 4, None, float("nan")], strict=False)
    in_memory_project.put("🐻‍❄️", polars_df)

    response = client.get("/api/project/activity")
    assert response.status_code == 200
    feed = response.json()
    assert len(feed[0]["value"]["data"]) == 6


def test_serialize_pandas_series_with_missing_values(client, in_memory_project):
    pandas_series = pandas.Series([1, 2, 3, 4, None, float("nan")])
    in_memory_project.put("🐼", pandas_series)

    response = client.get("/api/project/activity")
    assert response.status_code == 200
    feed = response.json()
    assert len(feed[0]["value"]) == 6


def test_serialize_polars_series_with_missing_values(client, in_memory_project):
    polars_df = polars.Series([1, 2, 3, 4, None, float("nan")], strict=False)
    in_memory_project.put("🐻‍❄️", polars_df)

    response = client.get("/api/project/activity")
    assert response.status_code == 200
    feed = response.json()
    assert len(feed[0]["value"]) == 6


def test_serialize_numpy_array(client, in_memory_project):
    np_array = numpy.array([1, 2, 3, 4])
    in_memory_project.put("np array", np_array)

    response = client.get("/api/project/activity")
    assert response.status_code == 200
    feed = response.json()
    assert len(feed[0]["value"]) == 4


def test_serialize_sklearn_estimator(client, in_memory_project):
    estimator = Lasso()
    in_memory_project.put("estimator", estimator)

    response = client.get("/api/project/activity")
    assert response.status_code == 200
    feed = response.json()
    assert feed[0]["value"] is not None


class FakeFigure(matplotlib.figure.Figure):
    def savefig(self, stream, *args, **kwargs):
        stream.write(b"<figure>")


def test_serialize_matplotlib_item(
    client,
    in_memory_project,
    monkeypatch_datetime,
    mock_nowstr,
):
    figure = FakeFigure()

    with io.BytesIO() as stream:
        figure.savefig(stream, format="svg", bbox_inches="tight")

        figure_bytes = stream.getvalue()
        figure_b64_str = base64.b64encode(figure_bytes).decode()

    in_memory_project.put("figure", figure)
    response = client.get("/api/project/activity")

    assert response.status_code == 200
    assert response.json() == [
        {
            "name": "figure",
            "media_type": "image/svg+xml;base64",
            "value": figure_b64_str,
            "updated_at": mock_nowstr,
            "created_at": mock_nowstr,
            "note": None,
            "version": 0,
        }
    ]


def test_serialize_altair_item(
    client,
    in_memory_project,
    monkeypatch_datetime,
    mock_nowstr,
):
    chart = altair.Chart().mark_point()
    chart_str = chart.to_json()
    chart_bytes = chart_str.encode("utf-8")
    chart_b64_str = base64.b64encode(chart_bytes).decode()

    in_memory_project.put("chart", chart)
    response = client.get("/api/project/activity")

    assert response.status_code == 200
    assert response.json() == [
        {
            "name": "chart",
            "media_type": "application/vnd.vega.v5+json;base64",
            "value": chart_b64_str,
            "updated_at": mock_nowstr,
            "created_at": mock_nowstr,
            "note": None,
            "version": 0,
        }
    ]


def test_serialize_pillow_item(
    client,
    in_memory_project,
    monkeypatch_datetime,
    mock_nowstr,
):
    image_array = numpy.random.rand(100, 100, 3) * 255
    image = Image.fromarray(image_array.astype("uint8")).convert("RGBA")

    with io.BytesIO() as stream:
        image.save(stream, format="png")

        png_bytes = stream.getvalue()
        png_b64_str = base64.b64encode(png_bytes).decode()

    in_memory_project.put("image", image)
    response = client.get("/api/project/activity")

    assert response.status_code == 200
    assert response.json() == [
        {
            "name": "image",
            "media_type": "image/png;base64",
            "value": png_b64_str,
            "updated_at": mock_nowstr,
            "created_at": mock_nowstr,
            "note": None,
            "version": 0,
        }
    ]


def test_serialize_plotly_item(
    client,
    in_memory_project,
    monkeypatch_datetime,
    mock_nowstr,
):
    bar = plotly.graph_objects.Bar(x=[1, 2, 3], y=[1, 3, 2])
    figure = plotly.graph_objects.Figure(data=[bar])
    figure_str = plotly.io.to_json(figure, engine="json")
    figure_bytes = figure_str.encode("utf-8")
    figure_b64_str = base64.b64encode(figure_bytes).decode()

    in_memory_project.put("figure", figure)
    response = client.get("/api/project/activity")

    assert response.status_code == 200
    assert response.json() == [
        {
            "name": "figure",
            "media_type": "application/vnd.plotly.v1+json;base64",
            "value": figure_b64_str,
            "updated_at": mock_nowstr,
            "created_at": mock_nowstr,
            "note": None,
            "version": 0,
        }
    ]


def test_serialize_primitive_item(
    client,
    in_memory_project,
    monkeypatch_datetime,
    mock_nowstr,
):
    in_memory_project.put("primitive", [1, 2, [3, 4]])
    response = client.get("/api/project/activity")

    assert response.status_code == 200
    assert response.json() == [
        {
            "name": "primitive",
            "media_type": "text/markdown",
            "value": [1, 2, [3, 4]],
            "updated_at": mock_nowstr,
            "created_at": mock_nowstr,
            "note": None,
            "version": 0,
        }
    ]


def test_serialize_primitive_item_with_nan(
    client,
    in_memory_project,
    monkeypatch_datetime,
    mock_nowstr,
):
    in_memory_project.put("primitive", float("nan"))
    response = client.get("/api/project/activity")

    assert response.status_code == 200
    assert response.json() == [
        {
            "name": "primitive",
            "media_type": "text/markdown",
            "value": None,
            "updated_at": mock_nowstr,
            "created_at": mock_nowstr,
            "note": None,
            "version": 0,
        }
    ]


def test_serialize_media_item(
    client,
    in_memory_project,
    monkeypatch_datetime,
    mock_nowstr,
):
    in_memory_project.put("media", "<media>", display_as="HTML")
    response = client.get("/api/project/activity")

    assert response.status_code == 200
    assert response.json() == [
        {
            "name": "media",
            "media_type": "text/html",
            "value": "<media>",
            "updated_at": mock_nowstr,
            "created_at": mock_nowstr,
            "note": None,
            "version": 0,
        }
    ]


def test_activity_feed(monkeypatch, client, in_memory_project):
    class MockDatetime:
        NOW = datetime.datetime.now(tz=datetime.timezone.utc)
        TIMEDELTA = datetime.timedelta(days=1)

        def __init__(self, *args, **kwargs): ...

        @staticmethod
        def now(*args, **kwargs):
            MockDatetime.NOW += MockDatetime.TIMEDELTA
            return MockDatetime.NOW

    monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)

    for i in range(5):
        in_memory_project.put(str(i), i)

    response = client.get("/api/project/activity")
    assert response.status_code == 200
    assert [(item["name"], item["value"]) for item in response.json()] == [
        ("4", 4),
        ("3", 3),
        ("2", 2),
        ("1", 1),
        ("0", 0),
    ]

    now = MockDatetime.NOW  # increments now

    in_memory_project.put("4", 5)
    in_memory_project.put("5", 5)

    response = client.get("/api/project/activity", params={"after": now})
    assert response.status_code == 200
    assert [(item["name"], item["value"]) for item in response.json()] == [
        ("5", 5),
        ("4", 5),
    ]


def test_get_items_with_pickle_item(
    monkeypatch,
    MockDatetime,
    mock_nowstr,
    client,
    in_memory_project,
):
    monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)
    in_memory_project.put("pickle", object)

    response = client.get("/api/project/activity")

    assert response.status_code == 200
    assert response.json() == [
        {
            "created_at": mock_nowstr,
            "updated_at": mock_nowstr,
            "name": "pickle",
            "media_type": "text/markdown",
            "value": "```python\n<class 'object'>\n```",
            "note": None,
            "version": 0,
        },
    ]


def test_get_items_with_pickle_item_and_unpickling_error(
    monkeypatch,
    MockDatetime,
    mock_nowstr,
    client,
    in_memory_project,
):
    import skore.persistence.item

    monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)
    in_memory_project.put("pickle", skore.persistence.item.NumpyArrayItem)

    monkeypatch.delattr("skore.persistence.item.numpy_array_item.NumpyArrayItem")
    monkeypatch.setattr(
        "skore.persistence.item.pickle_item.format_exception",
        lambda *args, **kwargs: "<traceback>",
    )

    response = client.get("/api/project/activity")

    assert response.status_code == 200
    assert response.json() == [
        {
            "created_at": mock_nowstr,
            "updated_at": mock_nowstr,
            "name": "pickle",
            "media_type": "text/markdown",
            "value": "Item cannot be displayed",
            "note": ("\n\nUnpicklingError with complete traceback:\n\n<traceback>"),
            "version": 0,
        },
    ]


def test_set_note(client, in_memory_project):
    for i in range(3):
        in_memory_project.put("notted", i)

    for i in range(3):
        response = client.put(
            "/api/project/note",
            json={
                "key": "notted",
                "message": f"note{i}",
                "version": i,
            },
        )
        assert response.status_code == 201

    for i in range(3):
        note = in_memory_project.get_note("notted", version=i)
        assert note == f"note{i}"
