import datetime

import numpy
import pandas
import polars
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from skore import CrossValidationReporter
from skore.item.media_item import MediaItem
from skore.ui.app import create_app
from skore.view.view import View


@pytest.fixture
def client(in_memory_project):
    return TestClient(app=create_app(project=in_memory_project))


def test_app_state(client):
    assert client.app.state.project is not None


def test_get_items(client, in_memory_project):
    response = client.get("/api/project/items")

    assert response.status_code == 200
    assert response.json() == {"views": {}, "items": {}}

    in_memory_project.put("test", "version_1")
    in_memory_project.put("test", "version_2")

    items = in_memory_project.get_item_versions("test")

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
    in_memory_project.put_view("hello", View(layout=[]))
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


def test_serialize_media_item(client, in_memory_project):
    imarray = numpy.random.rand(100, 100, 3) * 255
    img = Image.fromarray(imarray.astype("uint8")).convert("RGBA")
    in_memory_project.put("img", img)

    html = "<h1>éàªªUœALDXIWDŸΩΩ</h1>"
    in_memory_project.put("html", html)
    in_memory_project.put_item(
        "media html", MediaItem.factory_str(html, media_type="text/html")
    )

    response = client.get("/api/project/items")
    assert response.status_code == 200
    project = response.json()
    assert "image" in project["items"]["img"][0]["media_type"]
    assert project["items"]["html"][0]["value"] == html
    assert project["items"]["media html"][0]["value"] == html


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


def test_serialize_cross_validation_item(
    client, in_memory_project, fake_cross_validate
):
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
    cv_ui = project["items"]["cv"][0]["value"]
    assert len(cv_ui["scalar_results"]) == 3
    assert len(cv_ui["tabular_results"]) == 1
    assert "plots" in cv_ui
    assert "sections" in cv_ui


def test_activity_feed(monkeypatch, client, in_memory_project):
    class MockDatetime:
        NOW = datetime.datetime.now(tz=datetime.timezone.utc)
        TIMEDELTA = datetime.timedelta(days=1)

        def __init__(self, *args, **kwargs): ...

        @staticmethod
        def now(*args, **kwargs):
            MockDatetime.NOW += MockDatetime.TIMEDELTA
            return MockDatetime.NOW

    monkeypatch.setattr("skore.item.item.datetime", MockDatetime)

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
