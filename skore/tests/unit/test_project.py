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
from skore.project import Project, ProjectLoadError, ProjectPutError, load
from skore.view.view import View


def test_put_string_item(in_memory_project):
    in_memory_project.put("string_item", "Hello, World!")
    assert in_memory_project.get("string_item") == "Hello, World!"


def test_put_int_item(in_memory_project):
    in_memory_project.put("int_item", 42)
    assert in_memory_project.get("int_item") == 42


def test_put_float_item(in_memory_project):
    in_memory_project.put("float_item", 3.14)
    assert in_memory_project.get("float_item") == 3.14


def test_put_bool_item(in_memory_project):
    in_memory_project.put("bool_item", True)
    assert in_memory_project.get("bool_item") is True


def test_put_list_item(in_memory_project):
    in_memory_project.put("list_item", [1, 2, 3])
    assert in_memory_project.get("list_item") == [1, 2, 3]


def test_put_dict_item(in_memory_project):
    in_memory_project.put("dict_item", {"key": "value"})
    assert in_memory_project.get("dict_item") == {"key": "value"}


def test_put_pandas_dataframe(in_memory_project):
    dataframe = pandas.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    in_memory_project.put("pandas_dataframe", dataframe)
    pandas.testing.assert_frame_equal(
        in_memory_project.get("pandas_dataframe"), dataframe
    )


def test_put_pandas_series(in_memory_project):
    series = pandas.Series([0, 1, 2])
    in_memory_project.put("pandas_series", series)
    pandas.testing.assert_series_equal(in_memory_project.get("pandas_series"), series)


def test_put_numpy_array(in_memory_project):
    # Add a Numpy array
    arr = numpy.array([1, 2, 3, 4, 5])
    in_memory_project.put("numpy_array", arr)  # NumpyArrayItem
    numpy.testing.assert_array_equal(in_memory_project.get("numpy_array"), arr)


def test_put_mpl_figure(in_memory_project, monkeypatch):
    # Add a Matplotlib figure
    def savefig(*args, **kwargs):
        return ""

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", savefig)
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4])

    in_memory_project.put("mpl_figure", fig)  # MediaItem (SVG)
    assert isinstance(in_memory_project.get("mpl_figure"), bytes)


def test_put_vega_chart(in_memory_project):
    # Add an Altair chart
    altair_chart = altair.Chart().mark_point()
    in_memory_project.put("vega_chart", altair_chart)
    assert isinstance(in_memory_project.get("vega_chart"), bytes)


def test_put_pil_image(in_memory_project):
    # Add a PIL Image
    pil_image = Image.new("RGB", (100, 100), color="red")
    with BytesIO() as output:
        # FIXME: Not JPEG!
        pil_image.save(output, format="jpeg")

    in_memory_project.put("pil_image", pil_image)  # MediaItem (PNG)
    assert isinstance(in_memory_project.get("pil_image"), bytes)


def test_put_rf_model(in_memory_project, monkeypatch):
    # Add a scikit-learn model
    monkeypatch.setattr("sklearn.utils.estimator_html_repr", lambda _: "")
    model = RandomForestClassifier()
    model.fit(numpy.array([[1, 2], [3, 4]]), [0, 1])
    in_memory_project.put("rf_model", model)  # ScikitLearnModelItem
    assert isinstance(in_memory_project.get("rf_model"), RandomForestClassifier)


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

    # Test without `.skore`
    os.chdir(tmp_path.parent)
    p = load(tmp_path.name)
    assert isinstance(p, Project)


def test_put(in_memory_project):
    in_memory_project.put("key1", 1)
    in_memory_project.put("key2", 2)
    in_memory_project.put("key3", 3)
    in_memory_project.put("key4", 4)

    assert in_memory_project.list_item_keys() == ["key1", "key2", "key3", "key4"]


def test_put_kwargs(in_memory_project):
    in_memory_project.put(key="key1", value=1)
    assert in_memory_project.list_item_keys() == ["key1"]


def test_put_wrong_key_type(in_memory_project):
    with pytest.raises(ProjectPutError):
        in_memory_project.put(key=2, value=1)
    assert in_memory_project.list_item_keys() == []


def test_put_twice(in_memory_project):
    in_memory_project.put("key2", 2)
    in_memory_project.put("key2", 5)

    assert in_memory_project.get("key2") == 5


def test_get(in_memory_project):
    in_memory_project.put("key1", 1)
    assert in_memory_project.get("key1") == 1

    with pytest.raises(KeyError):
        in_memory_project.get("key2")


def test_delete(in_memory_project):
    in_memory_project.put("key1", 1)
    in_memory_project.delete_item("key1")

    assert in_memory_project.list_item_keys() == []

    with pytest.raises(KeyError):
        in_memory_project.delete_item("key2")


def test_keys(in_memory_project):
    in_memory_project.put("key1", 1)
    in_memory_project.put("key2", 2)
    assert in_memory_project.list_item_keys() == ["key1", "key2"]


def test_view(in_memory_project):
    layout = ["key1", "key2"]

    view = View(layout=layout)

    in_memory_project.put_view("view", view)
    assert in_memory_project.get_view("view") == view


def test_list_view_keys(in_memory_project):
    view = View(layout=[])

    in_memory_project.put_view("view", view)
    assert in_memory_project.list_view_keys() == ["view"]


def test_put_several_happy_path(in_memory_project):
    in_memory_project.put({"a": "foo", "b": "bar"})
    assert in_memory_project.list_item_keys() == ["a", "b"]


def test_put_several_some_errors(in_memory_project):
    with pytest.raises(ProjectPutError):
        in_memory_project.put({0: "hello", 1: "hello", 2: "hello"})
    assert in_memory_project.list_item_keys() == []


def test_put_several_nested(in_memory_project):
    in_memory_project.put({"a": {"b": "baz"}})
    assert in_memory_project.list_item_keys() == ["a"]
    assert in_memory_project.get("a") == {"b": "baz"}


def test_put_several_error(in_memory_project):
    """If some key-value pairs are wrong, add all that are valid and print a warning."""
    with pytest.raises(ProjectPutError):
        in_memory_project.put({"a": "foo", "b": (lambda: "unsupported object")})
    assert in_memory_project.list_item_keys() == ["a"]


def test_put_key_is_a_tuple(in_memory_project):
    """If key is not a string, warn."""
    with pytest.raises(ProjectPutError):
        in_memory_project.put(("a", "foo"), ("b", "bar"))
    assert in_memory_project.list_item_keys() == []


def test_put_key_is_a_set(in_memory_project):
    """Cannot use an unhashable type as a key."""
    with pytest.raises(ProjectPutError):
        in_memory_project.put(set(), "hello")


def test_put_wrong_key_and_value_raise(in_memory_project):
    """When `on_error` is "raise", raise the first error that occurs."""
    with pytest.raises(ProjectPutError):
        in_memory_project.put(0, (lambda: "unsupported object"))
