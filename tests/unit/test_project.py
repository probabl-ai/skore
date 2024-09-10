import json

import altair
import numpy
import numpy.testing
import pandas
import pandas.testing
import sklearn.svm
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from skore.project import Item, ItemType, Project, deserialize, serialize
from skore.storage.non_persistent_storage import NonPersistentStorage


def test_transform_primitive():
    o = 3
    actual = serialize(o)
    expected = Item(raw=3, item_type=ItemType.JSON, serialized="3")
    assert actual == expected


def test_transform_pandas_dataframe():
    o = pandas.DataFrame([{"key": "value"}])
    actual = serialize(o)
    expected = Item(
        raw=o,
        item_type=ItemType.PANDAS_DATAFRAME,
        serialized=o.to_json(orient="split"),
    )

    assert actual == expected


def test_transform_numpy_ndarray():
    o = numpy.array([1, 2, 3])
    actual = serialize(o)
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
    actual = serialize(o)
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

    s = serialize(fig)

    assert s.media_type == "image/svg+xml"
    assert s.serialized == ""


def test_untransform_primitive():
    o = 3
    transformed = serialize(o)
    assert (
        deserialize(
            serialized=transformed.serialized,
            item_type=transformed.item_type,
            media_type=None,
        ).raw
        == o
    )


def test_untransform_pandas_dataframe():
    o = pandas.DataFrame([{"key": "value"}])
    item = deserialize(
        serialized=o.to_json(orient="split"),
        item_type=ItemType.PANDAS_DATAFRAME,
        media_type=None,
    )

    pandas.testing.assert_frame_equal(item.raw, o)


def test_untransform_numpy_ndarray():
    o = numpy.array([1, 2, 3])
    item = deserialize(
        serialized=json.dumps(o.tolist()),
        item_type=ItemType.NUMPY_ARRAY,
        media_type=None,
    )

    numpy.testing.assert_array_equal(o, item.raw)


def test_untransform_sklearn_model():
    o = sklearn.svm.SVC()
    t = serialize(o)
    u = deserialize(serialized=t.serialized, item_type=t.item_type, media_type=None)

    assert isinstance(u.raw, sklearn.svm.SVC)


def test_untransform_matplotlib_figure(monkeypatch):
    def savefig(*args, **kwargs):
        return ""

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", savefig)

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4])

    s = serialize(fig)

    assert deserialize(
        serialized=s.serialized, item_type=s.item_type, media_type=s.media_type
    ) == Item(
        raw=None,
        item_type=ItemType.MEDIA,
        serialized="",
        media_type="image/svg+xml",
    )


def test_project_here(monkeypatch):
    def savefig(*args, **kwargs):
        return ""

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", savefig)
    monkeypatch.setattr("sklearn.utils.estimator_html_repr", lambda _: "")
    monkeypatch.setattr("skops.io.dumps", lambda _: b"")

    project = Project(NonPersistentStorage())
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
    project.put("vega_chart", altair.Chart().mark_point())

    # Add a PIL Image
    # pil_image = Image.new("RGB", (100, 100), color="red")
    # project.put("pil_image", pil_image)  # MediaItem (PNG)

    # Add raw bytes with media type
    # raw_bytes = b"Some raw data"
    # project.put_item("raw_data", MediaItem(raw_bytes, "application/octet-stream"))

    # Add a scikit-learn model
    model = RandomForestClassifier()
    model.fit(numpy.array([[1, 2], [3, 4]]), [0, 1])
    project.put("rf_model", model)  # ScikitLearnModelItem

    assert project.storage.content == {
        "string_item": {
            "item_type": str(ItemType.JSON),
            "serialized": '"Hello, World!"',
            "media_type": None,
        },
        "int_item": {
            "item_type": str(ItemType.JSON),
            "serialized": "42",
            "media_type": None,
        },
        "float_item": {
            "item_type": str(ItemType.JSON),
            "serialized": "3.14",
            "media_type": None,
        },
        "bool_item": {
            "item_type": str(ItemType.JSON),
            "serialized": "true",
            "media_type": None,
        },
        "list_item": {
            "item_type": str(ItemType.JSON),
            "serialized": "[1, 2, 3]",
            "media_type": None,
        },
        "dict_item": {
            "item_type": str(ItemType.JSON),
            "serialized": '{"key": "value"}',
            "media_type": None,
        },
        "pandas_df": {
            "item_type": str(ItemType.PANDAS_DATAFRAME),
            "serialized": '{"columns":["A","B"],"index":[0,1,2],"data":[[1,4],[2,5],[3,6]]}',
            "media_type": None,
        },
        "numpy_array": {
            "item_type": str(ItemType.NUMPY_ARRAY),
            "serialized": "[1, 2, 3, 4, 5]",
            "media_type": None,
        },
        "mpl_figure": {
            "item_type": str(ItemType.MEDIA),
            "serialized": "",
            "media_type": "image/svg+xml",
        },
        "rf_model": {
            "item_type": str(ItemType.SKLEARN_BASE_ESTIMATOR),
            "serialized": '{"skops": "", "html": ""}',
            "media_type": "text/html",
        },
        "vega_chart": {
            "item_type": str(ItemType.ALTAIR_CHART),
            "serialized": "",
            "media_type": None,
        },
    }

    assert project.get("string_item") == "Hello, World!"
    assert project.get("int_item") == 42
    assert project.get("float_item") == 3.14
    assert project.get("bool_item")
    assert project.get("list_item") == [1, 2, 3]
    assert project.get("dict_item") == {"key": "value"}


def test_api_get_items():
    from fastapi.testclient import TestClient
    from skore.api import create_api_app

    project = Project(NonPersistentStorage())
    project.put("string_item", "Hello, World!")
    project.put("int_item", 42)

    client = TestClient(app=create_api_app(project))
    response = client.get("/api/skores")
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
