import base64
import datetime
import io
import json

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
from sklearn.model_selection import KFold
from skore import CrossValidationReporter
from skore.persistence.view.view import View
from skore.ui.app import create_app


@pytest.fixture
def client(in_memory_project):
    return TestClient(app=create_app(project=in_memory_project))


@pytest.fixture
def monkeypatch_datetime(monkeypatch, MockDatetime):
    monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)


def test_app_state(client):
    assert client.app.state.project is not None


def test_get_items(client, in_memory_project):
    response = client.get("/api/project/items")

    assert response.status_code == 200
    assert response.json() == {"views": {}, "items": {}}

    in_memory_project.put("test", "version_1")
    in_memory_project.put("test", "version_2")

    items = in_memory_project.item_repository.get_item_versions("test")

    response = client.get("/api/project/items")
    assert response.status_code == 200
    assert response.json() == {
        "views": {},
        "items": {
            "test": [
                {
                    "name": "test",
                    "media_type": "text/markdown",
                    "value": item.primitive,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at,
                }
                for item in items
            ],
        },
    }


def test_put_view_layout(client):
    response = client.put("/api/project/views?key=hello", json=["test"])
    assert response.status_code == 201


def test_delete_view(client, in_memory_project):
    in_memory_project.view_repository.put_view("hello", View(layout=[]))
    response = client.delete("/api/project/views?key=hello")
    assert response.status_code == 202


def test_delete_view_missing(client):
    response = client.delete("/api/project/views?key=hello")
    assert response.status_code == 404


def test_serialize_pandas_dataframe_with_missing_values(client, in_memory_project):
    pandas_df = pandas.DataFrame([1, 2, 3, 4, None, float("nan")])
    in_memory_project.put("🐼", pandas_df)

    response = client.get("/api/project/items")
    assert response.status_code == 200
    project = response.json()
    assert len(project["items"]["🐼"][0]["value"]["data"]) == 6


def test_serialize_polars_dataframe_with_missing_values(client, in_memory_project):
    polars_df = polars.DataFrame([1, 2, 3, 4, None, float("nan")], strict=False)
    in_memory_project.put("🐻‍❄️", polars_df)

    response = client.get("/api/project/items")
    assert response.status_code == 200
    project = response.json()
    assert len(project["items"]["🐻‍❄️"][0]["value"]["data"]) == 6


def test_serialize_pandas_series_with_missing_values(client, in_memory_project):
    pandas_series = pandas.Series([1, 2, 3, 4, None, float("nan")])
    in_memory_project.put("🐼", pandas_series)

    response = client.get("/api/project/items")
    assert response.status_code == 200
    project = response.json()
    assert len(project["items"]["🐼"][0]["value"]) == 6


def test_serialize_polars_series_with_missing_values(client, in_memory_project):
    polars_df = polars.Series([1, 2, 3, 4, None, float("nan")], strict=False)
    in_memory_project.put("🐻‍❄️", polars_df)

    response = client.get("/api/project/items")
    assert response.status_code == 200
    project = response.json()
    assert len(project["items"]["🐻‍❄️"][0]["value"]) == 6


def test_serialize_numpy_array(client, in_memory_project):
    np_array = numpy.array([1, 2, 3, 4])
    in_memory_project.put("np array", np_array)

    response = client.get("/api/project/items")
    assert response.status_code == 200
    project = response.json()
    assert len(project["items"]["np array"][0]["value"]) == 4


def test_serialize_sklearn_estimator(client, in_memory_project):
    estimator = Lasso()
    in_memory_project.put("estimator", estimator)

    response = client.get("/api/project/items")
    assert response.status_code == 200
    project = response.json()
    assert project["items"]["estimator"][0]["value"] is not None


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
        figure_bytes_b64 = base64.b64encode(figure_bytes).decode()

    in_memory_project.put("figure", figure)
    response = client.get("/api/project/items")

    assert response.status_code == 200
    assert response.json() == {
        "views": {},
        "items": {
            "figure": [
                {
                    "name": "figure",
                    "media_type": "image/svg+xml;base64",
                    "value": figure_bytes_b64,
                    "updated_at": mock_nowstr,
                    "created_at": mock_nowstr,
                }
            ]
        },
    }


def test_serialize_altair_item(
    client,
    in_memory_project,
    monkeypatch_datetime,
    mock_nowstr,
):
    chart = altair.Chart().mark_point()
    chart_str = chart.to_json()
    chart_bytes = chart_str.encode("utf-8")
    chart_bytes_b64 = base64.b64encode(chart_bytes).decode()

    in_memory_project.put("chart", chart)
    response = client.get("/api/project/items")

    assert response.status_code == 200
    assert response.json() == {
        "views": {},
        "items": {
            "chart": [
                {
                    "name": "chart",
                    "media_type": "application/vnd.vega.v5+json;base64",
                    "value": chart_bytes_b64,
                    "updated_at": mock_nowstr,
                    "created_at": mock_nowstr,
                }
            ]
        },
    }


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
        png_bytes_b64 = base64.b64encode(png_bytes).decode()

    in_memory_project.put("image", image)
    response = client.get("/api/project/items")

    assert response.status_code == 200
    assert response.json() == {
        "views": {},
        "items": {
            "image": [
                {
                    "name": "image",
                    "media_type": "image/png;base64",
                    "value": png_bytes_b64,
                    "updated_at": mock_nowstr,
                    "created_at": mock_nowstr,
                }
            ]
        },
    }


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
    figure_bytes_b64 = base64.b64encode(figure_bytes).decode()

    in_memory_project.put("figure", figure)
    response = client.get("/api/project/items")

    assert response.status_code == 200
    assert response.json() == {
        "views": {},
        "items": {
            "figure": [
                {
                    "name": "figure",
                    "media_type": "application/vnd.plotly.v1+json;base64",
                    "value": figure_bytes_b64,
                    "updated_at": mock_nowstr,
                    "created_at": mock_nowstr,
                }
            ]
        },
    }


def test_serialize_media_item(client, in_memory_project): ...


@pytest.fixture
def fake_cross_validate(monkeypatch):
    def _fake_cross_validate(*args, **kwargs):
        result = {
            "test_score": numpy.array([1.0] * 5),
            "score_time": numpy.array([1.0] * 5),
            "fit_time": numpy.array([1.0] * 5),
        }
        if kwargs.get("return_estimator"):
            result["estimator"] = numpy.array([])
        if kwargs.get("return_indices"):
            result["indices"] = {
                "train": numpy.array([[1.0] * 5] * 5),
                "test": numpy.array([[1.0] * 5] * 5),
            }
        if kwargs.get("return_train_score"):
            result["train_score"] = numpy.array([1.0] * 5)
        return result

    monkeypatch.setattr("sklearn.model_selection.cross_validate", _fake_cross_validate)


def test_serialize_cross_validation_reporter_item(
    client,
    in_memory_project,
    monkeypatch,
    MockDatetime,
    mock_nowstr,
    fake_cross_validate,
):
    monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)
    monkeypatch.setattr(
        "skore.sklearn.cross_validation.cross_validation_reporter.plot_cross_validation_compare_scores",
        lambda _: {},
    )
    monkeypatch.setattr(
        "skore.sklearn.cross_validation.cross_validation_reporter.plot_cross_validation_timing",
        lambda _: {},
    )

    def prepare_cv():
        from sklearn import datasets, linear_model

        diabetes = datasets.load_diabetes()
        X = diabetes.data[:100]
        y = diabetes.target[:100]
        lasso = linear_model.Lasso(alpha=2.5)
        return lasso, X, y

    model, X, y = prepare_cv()
    reporter = CrossValidationReporter(model, X, y, cv=KFold(3))
    in_memory_project.put("cv", reporter)

    response = client.get("/api/project/items")
    assert response.status_code == 200

    project = response.json()
    expected = {
        "name": "cv",
        "media_type": "application/vnd.skore.cross_validation+json",
        "value": {
            "scalar_results": [
                {
                    "name": "Mean test score",
                    "value": 1.0,
                    "stddev": 0.0,
                    "favorability": "greater_is_better",
                },
                {
                    "name": "Mean score time (seconds)",
                    "value": 1.0,
                    "stddev": 0.0,
                    "favorability": "lower_is_better",
                },
                {
                    "name": "Mean fit time (seconds)",
                    "value": 1.0,
                    "stddev": 0.0,
                    "favorability": "lower_is_better",
                },
            ],
            "tabular_results": [
                {
                    "name": "Cross validation results",
                    "columns": ["test_score", "score_time", "fit_time"],
                    "data": [
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ],
                    "favorability": [
                        "greater_is_better",
                        "lower_is_better",
                        "lower_is_better",
                    ],
                }
            ],
            "plots": [
                {
                    "name": "Scores",
                    "value": json.loads(plotly.io.to_json({}, engine="json")),
                },
                {
                    "name": "Timings",
                    "value": json.loads(plotly.io.to_json({}, engine="json")),
                },
            ],
            "sections": [
                {
                    "title": "Model",
                    "icon": "icon-square-cursor",
                    "items": [
                        {
                            "name": "Estimator parameters",
                            "description": "Core model configuration used for training",
                            "value": (
                                '<a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html" target="_blank"><code>Lasso</code></a>\n'  # noqa: E501
                                "- alpha: *2.5*\n"
                                "- copy_X: True (default)\n"
                                "- fit_intercept: True (default)\n"
                                "- max_iter: 1000 (default)\n"
                                "- positive: False (default)\n"
                                "- precompute: False (default)\n"
                                "- random_state: None (default)\n"
                                "- selection: 'cyclic' (default)\n"
                                "- tol: 0.0001 (default)\n"
                                "- warm_start: False (default)"
                            ),
                        },
                        {
                            "name": "Cross-validation parameters",
                            "description": "Controls how data is split and validated",
                            "value": (
                                "n_splits: *3*, "
                                "shuffle: *False*, "
                                "random_state: *None*"
                            ),
                        },
                    ],
                }
            ],
        },
        "updated_at": mock_nowstr,
        "created_at": mock_nowstr,
    }
    actual = project["items"]["cv"][0]
    assert expected == actual


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

    response = client.get("/api/project/items")

    assert response.status_code == 200
    assert response.json() == {
        "items": {
            "pickle": [
                {
                    "created_at": mock_nowstr,
                    "updated_at": mock_nowstr,
                    "name": "pickle",
                    "media_type": "text/markdown",
                    "value": "<class 'object'>",
                },
            ],
        },
        "views": {},
    }
