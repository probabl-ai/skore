import os
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import altair
import numpy
import numpy.testing
import pandas
import pandas.testing
import pytest
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from skore.persistence.memory import InMemoryStorage
from skore.project import Project, ProjectDoesNotExist, load


@pytest.fixture
def storage():
    return InMemoryStorage()


@pytest.fixture
def project(storage):
    return Project(storage)


# FIXME: Split this test into several small test
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


def test_load(tmp_path):
    with pytest.raises(ProjectDoesNotExist):
        load("/empty")

    # Project path must end with ".skore"
    project_path = tmp_path.parent / (tmp_path.name + ".skore")
    os.mkdir(project_path)
    load(project_path)

    with tempfile.TemporaryDirectory(dir=Path.cwd(), suffix=".skore") as tmp_dir:
        load(Path(tmp_dir).stem)


def test_put(project):
    project.put("key1", 1)
    project.put("key2", 2)
    project.put("key3", 3)
    project.put("key4", 4)

    assert project.list_keys() == ["key1", "key2", "key3", "key4"]


def test_put_twice(project):
    project.put("key2", 2)
    project.put("key2", 5)

    assert project.get("key2") == 5


def test_get(project):
    project.put("key1", 1)
    assert project.get("key1") == 1

    with pytest.raises(KeyError):
        project.get("key2")


def test_delete(project):
    project.put("key1", 1)
    project.delete_item("key1")

    assert project.list_keys() == []

    with pytest.raises(KeyError):
        project.delete_item("key2")


def test_keys(project):
    project.put("key1", 1)
    project.put("key2", 2)
    assert project.list_keys() == ["key1", "key2"]


def test_item_metadata():
    project = Project(InMemoryStorage())
    project.put("key", "hello")

    now = datetime.now(tz=timezone.utc)
    item = project.get_item("key")

    assert (now - item.created_at).seconds < 1
    assert item.updated_at == item.created_at

    previous_updated_at = item.updated_at
    project.put("key", "world")
    item = project.get_item("key")
    assert item.updated_at > previous_updated_at
