import os
from io import BytesIO

import altair
import numpy
import numpy.testing
import pandas
import pandas.testing
import pytest
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from skore.item import ItemRepository
from skore.layout import LayoutRepository
from skore.layout.layout import LayoutItem, LayoutItemSize
from skore.persistence.memory import InMemoryStorage
from skore.project import Project, ProjectDoesNotExist, load


@pytest.fixture
def project():
    return Project(
        item_repository=ItemRepository(InMemoryStorage()),
        layout_repository=LayoutRepository(InMemoryStorage()),
    )


def test_put_string_item(project):
    project.put("string_item", "Hello, World!")  # JSONItem
    assert project.get("string_item") == "Hello, World!"


def test_put_int_item(project):
    project.put("int_item", 42)  # JSONItem
    assert project.get("int_item") == 42


def test_put_float_item(project):
    project.put("float_item", 3.14)  # JSONItem
    assert project.get("float_item") == 3.14


def test_put_bool_item(project):
    project.put("bool_item", True)  # JSONItem
    assert project.get("bool_item") is True


def test_put_list_item(project):
    project.put("list_item", [1, 2, 3])  # JSONItem
    assert project.get("list_item") == [1, 2, 3]


def test_put_dict_item(project):
    project.put("dict_item", {"key": "value"})  # JSONItem
    assert project.get("dict_item") == {"key": "value"}


def test_put_pandas_df(project):
    df = pandas.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    project.put("pandas_df", df)  # DataFrameItem
    pandas.testing.assert_frame_equal(project.get("pandas_df"), df)


def test_put_numpy_array(project):
    # Add a Numpy array
    arr = numpy.array([1, 2, 3, 4, 5])
    project.put("numpy_array", arr)  # NumpyArrayItem
    numpy.testing.assert_array_equal(project.get("numpy_array"), arr)


def test_put_mpl_figure(project, monkeypatch):
    # Add a Matplotlib figure
    def savefig(*args, **kwargs):
        return ""

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", savefig)
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4])

    project.put("mpl_figure", fig)  # MediaItem (SVG)
    assert isinstance(project.get("mpl_figure"), bytes)


def test_put_vega_chart(project):
    # Add an Altair chart
    altair_chart = altair.Chart().mark_point()
    project.put("vega_chart", altair_chart)
    assert isinstance(project.get("vega_chart"), bytes)


def test_put_pil_image(project):
    # Add a PIL Image
    pil_image = Image.new("RGB", (100, 100), color="red")
    with BytesIO() as output:
        # FIXME: Not JPEG!
        pil_image.save(output, format="jpeg")

    project.put("pil_image", pil_image)  # MediaItem (PNG)
    assert isinstance(project.get("pil_image"), bytes)


def test_put_rf_model(project, monkeypatch):
    # Add a scikit-learn model
    monkeypatch.setattr("sklearn.utils.estimator_html_repr", lambda _: "")
    model = RandomForestClassifier()
    model.fit(numpy.array([[1, 2], [3, 4]]), [0, 1])
    project.put("rf_model", model)  # ScikitLearnModelItem
    assert isinstance(project.get("rf_model"), RandomForestClassifier)


def test_load(tmp_path):
    with pytest.raises(ProjectDoesNotExist):
        load("/empty")

    # Project path must end with ".skore"
    project_path = tmp_path.parent / (tmp_path.name + ".skore")
    os.mkdir(project_path)
    os.mkdir(project_path / "items")
    os.mkdir(project_path / "layouts")
    p = load(project_path)
    assert isinstance(p, Project)


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


def test_report_layout(project):
    layout = [
        LayoutItem(key="key1", size=LayoutItemSize.LARGE),
        LayoutItem(key="key2", size=LayoutItemSize.SMALL),
    ]

    project.put_report_layout(layout)
    assert project.get_report_layout() == layout
