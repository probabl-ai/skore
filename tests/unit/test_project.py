import json
import os
import tempfile
from io import BytesIO
from pathlib import Path

import altair
import numpy
import numpy.testing
import pandas
import pandas.testing
import pytest
import sklearn.svm
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from skore.persistence.memory import InMemoryStorage
from skore.project import (
    Item,
    ItemType,
    PersistedItem,
    Project,
    ProjectDoesNotExist,
    load,
    object_to_item,
    unpersist,
)


def test_transform_primitive():
    o = 3
    actual = object_to_item(o)
    expected = Item(raw=3, item_type=ItemType.JSON, serialized="3")
    assert actual == expected


def test_transform_pandas_dataframe():
    o = pandas.DataFrame([{"key": "value"}])
    actual = object_to_item(o)
    expected = Item(
        raw=o,
        item_type=ItemType.PANDAS_DATAFRAME,
        serialized=o.to_json(orient="split"),
    )

    assert actual == expected


def test_transform_numpy_ndarray():
    o = numpy.array([1, 2, 3])
    actual = object_to_item(o)
    expected = Item(
        raw=o,
        item_type=ItemType.NUMPY_ARRAY,
        serialized=json.dumps(o.tolist()),
    )

    assert actual == expected


def test_transform_sklearn_base_baseestimator(monkeypatch):
    monkeypatch.setattr("sklearn.utils.estimator_html_repr", lambda _: "")
    monkeypatch.setattr("skops.io.dumps", lambda _: b"")

    o = sklearn.svm.SVC()
    actual = object_to_item(o)
    expected = Item(
        raw=o,
        item_type=ItemType.SKLEARN_BASE_ESTIMATOR,
        serialized=json.dumps(
            {
                "skops": "",
                "html": "",
            }
        ),
        media_type="text/html",
    )

    assert actual == expected


def test_matplotlib(monkeypatch):
    def savefig(*args, **kwargs):
        return ""

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", savefig)

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4])

    s = object_to_item(fig)

    assert s.media_type == "image/svg+xml"
    assert s.serialized == ""


def test_untransform_primitive():
    o = 3
    item = object_to_item(o)

    expected = unpersist(
        PersistedItem(
            serialized=item.serialized,
            item_type=item.item_type,
            media_type=None,
        )
    ).raw

    assert o == expected


def test_untransform_pandas_dataframe():
    o = pandas.DataFrame([{"key": "value"}])
    item = unpersist(
        PersistedItem(
            serialized=o.to_json(orient="split"),
            item_type=ItemType.PANDAS_DATAFRAME,
            media_type=None,
        )
    )

    pandas.testing.assert_frame_equal(item.raw, o)


def test_untransform_numpy_ndarray():
    o = numpy.array([1, 2, 3])
    item = unpersist(
        PersistedItem(
            serialized=json.dumps(o.tolist()),
            item_type=ItemType.NUMPY_ARRAY,
            media_type=None,
        )
    )

    numpy.testing.assert_array_equal(o, item.raw)


def test_untransform_sklearn_model():
    o = sklearn.svm.SVC()
    item: Item = object_to_item(o)
    u: Item = unpersist(
        PersistedItem(
            serialized=item.serialized, item_type=item.item_type, media_type=None
        )
    )

    assert isinstance(u.raw, sklearn.svm.SVC)


def test_untransform_matplotlib_figure(monkeypatch):
    def savefig(*args, **kwargs):
        return ""

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", savefig)

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4])

    s: Item = object_to_item(fig)

    assert unpersist(
        PersistedItem(
            serialized=s.serialized, item_type=s.item_type, media_type=s.media_type
        )
    ) == Item(
        raw=None,
        item_type=ItemType.MEDIA,
        serialized="",
        media_type="image/svg+xml",
    )


def test_project(monkeypatch):
    def savefig(*args, **kwargs):
        return ""

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", savefig)
    monkeypatch.setattr("sklearn.utils.estimator_html_repr", lambda _: "")

    project = Project(InMemoryStorage())
    project.put("string_item", "Hello, World!")  # JSONItem
    project.put("int_item", 42)  # JSONItem
    project.put("float_item", 3.14)  # JSONItem
    project.put("bool_item", True)  # JSONItem
    project.put("list_item", [1, 2, 3])  # JSONItem
    project.put("dict_item", {"key": "value"})  # JSONItem

    # Add a DataFrame
    df = pandas.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    project.put("pandas_df", df)  # DataFrameItem

    # Add a Numpy array
    arr = numpy.array([1, 2, 3, 4, 5])
    project.put("numpy_array", arr)  # NumpyArrayItem

    # Add a Matplotlib figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4])
    project.put("mpl_figure", fig)  # MediaItem (SVG)

    # Add an Altair chart
    altair_chart = altair.Chart().mark_point()
    project.put("vega_chart", altair_chart)

    # Add a PIL Image
    pil_image = Image.new("RGB", (100, 100), color="red")
    project.put("pil_image", pil_image)  # MediaItem (PNG)

    with BytesIO() as output:
        pil_image.save(output, format="jpeg")

    # Add a scikit-learn model
    model = RandomForestClassifier()
    model.fit(numpy.array([[1, 2], [3, 4]]), [0, 1])
    project.put("rf_model", model)  # ScikitLearnModelItem

    assert project.get("string_item") == "Hello, World!"
    assert project.get("int_item") == 42
    assert project.get("float_item") == 3.14
    assert project.get("bool_item") is True
    assert project.get("list_item") == [1, 2, 3]
    assert project.get("dict_item") == {"key": "value"}
    pandas.testing.assert_frame_equal(project.get("pandas_df"), df)
    numpy.testing.assert_array_equal(project.get("numpy_array"), arr)
    assert isinstance(project.get("rf_model"), RandomForestClassifier)
    assert project.get("mpl_figure") is None
    assert project.get("vega_chart") is None
    assert project.get("pil_image") is None


def test_api_get_items():
    from fastapi.testclient import TestClient
    from skore.ui.app import create_app

    project = Project(InMemoryStorage())
    project.put("string_item", "Hello, World!")
    project.put("int_item", 42)

    client = TestClient(app=create_app(project))
    response = client.get("/api/items")
    json_response = response.json()

    assert response.status_code == 200

    assert json_response["string_item"] == {
        "item_type": ItemType.JSON,
        "media_type": None,
        "serialized": "Hello, World!",
    }
    assert json_response["int_item"] == {
        "item_type": ItemType.JSON,
        "media_type": None,
        "serialized": 42,
    }


def test_load(tmp_path):
    with pytest.raises(ProjectDoesNotExist):
        load("/empty")

    # Project path must end with ".skore"
    project_path = tmp_path.parent / (tmp_path.name + ".skore")
    os.mkdir(project_path)
    load(project_path)

    with tempfile.TemporaryDirectory(dir=Path.cwd(), suffix=".skore") as tmp_dir:
        load(Path(tmp_dir).stem)


class TestProject:
    @pytest.fixture
    def storage(self):
        return InMemoryStorage()

    @pytest.fixture
    def project(self, storage):
        return Project(storage)

    def test_put(self, project):
        project.put("key1", 1)
        project.put("key2", 2)
        project.put("key3", 3)
        project.put("key4", 4)

        assert project.list_keys() == ["key1", "key2", "key3", "key4"]

    def test_put_twice(self, project):
        project.put("key2", 2)
        project.put("key2", 5)

        assert project.get("key2") == 5

    def test_get(self, project):
        project.put("key1", 1)
        assert project.get("key1") == 1

        with pytest.raises(KeyError):
            project.get("key2")

    def test_delete(self, project):
        project.put("key1", 1)
        project.delete_item("key1")

        assert project.list_keys() == []

        with pytest.raises(KeyError):
            project.delete_item("key2")

    def test_keys(self, project):
        project.put("key1", 1)
        project.put("key2", 2)
        assert project.list_keys() == ["key1", "key2"]
