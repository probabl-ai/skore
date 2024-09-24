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
from skore.persistence.in_memory_storage import InMemoryStorage
from skore.project import Project, ProjectLoadError, ProjectPutError, load
from skore.view.view import LayoutItem, LayoutItemSize, View
from skore.view.view_repository import ViewRepository


@pytest.fixture
def project():
    return Project(
        item_repository=ItemRepository(InMemoryStorage()),
        view_repository=ViewRepository(InMemoryStorage()),
    )


def test_put_string_item(project):
    project.put("string_item", "Hello, World!")
    assert project.get("string_item") == "Hello, World!"


def test_put_int_item(project):
    project.put("int_item", 42)
    assert project.get("int_item") == 42


def test_put_float_item(project):
    project.put("float_item", 3.14)
    assert project.get("float_item") == 3.14


def test_put_bool_item(project):
    project.put("bool_item", True)
    assert project.get("bool_item") is True


def test_put_list_item(project):
    project.put("list_item", [1, 2, 3])
    assert project.get("list_item") == [1, 2, 3]


def test_put_dict_item(project):
    project.put("dict_item", {"key": "value"})
    assert project.get("dict_item") == {"key": "value"}


def test_put_pandas_dataframe(project):
    dataframe = pandas.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    project.put("pandas_dataframe", dataframe)
    pandas.testing.assert_frame_equal(project.get("pandas_dataframe"), dataframe)


def test_put_pandas_series(project):
    series = pandas.Series([0, 1, 2])
    project.put("pandas_series", series)
    pandas.testing.assert_series_equal(project.get("pandas_series"), series)


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
    with pytest.raises(ProjectLoadError):
        load("/empty")

    # Project path must end with ".skore"
    project_path = tmp_path.parent / (tmp_path.name + ".skore")
    os.mkdir(project_path)
    os.mkdir(project_path / "items")
    os.mkdir(project_path / "views")
    p = load(project_path)
    assert isinstance(p, Project)


def test_put(project):
    project.put("key1", 1)
    project.put("key2", 2)
    project.put("key3", 3)
    project.put("key4", 4)

    assert project.list_item_keys() == ["key1", "key2", "key3", "key4"]


def test_put_twice(project):
    project.put("key2", 2)
    project.put("key2", 5)

    assert project.get("key2") == 5


def test_put_int_key(project, caplog):
    # Warns that 0 is not a string, but doesn't raise
    project.put(0, "hello")
    assert len(caplog.record_tuples) == 1
    assert project.list_item_keys() == []


def test_get(project):
    project.put("key1", 1)
    assert project.get("key1") == 1

    with pytest.raises(KeyError):
        project.get("key2")


def test_delete(project):
    project.put("key1", 1)
    project.delete_item("key1")

    assert project.list_item_keys() == []

    with pytest.raises(KeyError):
        project.delete_item("key2")


def test_keys(project):
    project.put("key1", 1)
    project.put("key2", 2)
    assert project.list_item_keys() == ["key1", "key2"]


def test_view(project):
    layout = [
        LayoutItem(key="key1", size=LayoutItemSize.LARGE),
        LayoutItem(key="key2", size=LayoutItemSize.SMALL),
    ]

    view = View(layout=layout)

    project.put_view("view", view)
    assert project.get_view("view") == view


def test_list_view_keys(project):
    view = View(layout=[])

    project.put_view("view", view)
    assert project.list_view_keys() == ["view"]


def test_put_several_happy_path(project):
    project.put({"a": "foo", "b": "bar"})
    assert project.list_item_keys() == ["a", "b"]


def test_put_several_canonical(project):
    """Use `put_several` instead of the `put` alias."""
    project.put_several({"a": "foo", "b": "bar"})
    assert project.list_item_keys() == ["a", "b"]


def test_put_several_some_errors(project, caplog):
    project.put(
        {
            0: "hello",
            1: "hello",
            2: "hello",
        }
    )
    assert len(caplog.record_tuples) == 3
    assert project.list_item_keys() == []


def test_put_several_nested(project):
    project.put({"a": {"b": "baz"}})
    assert project.list_item_keys() == ["a"]
    assert project.get("a") == {"b": "baz"}


def test_put_several_error(project):
    """If some key-value pairs are wrong, add all that are valid and print a warning."""
    project.put({"a": "foo", "b": (lambda: "unsupported object")})
    assert project.list_item_keys() == ["a"]


def test_put_key_is_a_tuple(project):
    """If key is not a string, warn."""
    project.put(("a", "foo"), ("b", "bar"))
    assert project.list_item_keys() == []


def test_put_key_is_a_set(project):
    """Cannot use an unhashable type as a key."""
    with pytest.raises(ProjectPutError):
        project.put(set(), "hello", on_error="raise")


def test_put_wrong_key_and_value_raise(project):
    """When `on_error` is "raise", raise the first error that occurs."""
    with pytest.raises(ProjectPutError):
        project.put(0, (lambda: "unsupported object"), on_error="raise")
