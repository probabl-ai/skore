import altair
import numpy
import numpy.testing
import pandas
import pandas.testing
import plotly
import polars
import polars.testing
import pytest
from matplotlib.pyplot import subplots
from matplotlib.testing.compare import compare_images
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from skore import Project
from skore.project.project import ProjectDeletedError


@pytest.fixture(autouse=True)
def monkeypatch_datetime(monkeypatch, MockDatetime):
    monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)


def test_init(tmp_path):
    dirpath = tmp_path / "my-project.skore"

    # Ensure missing project can be created
    Project(dirpath)

    assert dirpath.exists()
    assert (dirpath / "items").exists()

    # Ensure existing project raises an error with `if_exists="raise"`
    with pytest.raises(FileExistsError):
        Project(dirpath, if_exists="raise")

    # Ensure existing project can be loaded with `if_exists="load"`
    Project(dirpath, if_exists="load")


def test_clear(tmp_path):
    dirpath = tmp_path / "my-project.skore"
    project = Project(dirpath)

    project.put("<key>", "<value>")

    assert project.keys() == ["<key>"]

    project.clear()

    assert project.keys() == []
    assert project._item_repository.keys() == []

    assert dirpath.exists()


def test_clear_delete_project(tmp_path):
    dirpath = tmp_path / "my-project.skore"
    project = Project(dirpath)

    project.clear(delete_project=True)
    assert not dirpath.exists()

    with pytest.raises(
        ProjectDeletedError, match="This Project instance is marked as deleted"
    ):
        project.keys()


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
    dataframe = pandas.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": [7, 8, 9],
        },
        index=pandas.Index([0, 1, 2], name="myIndex"),
    )

    in_memory_project.put("pandas_dataframe", dataframe)
    pandas.testing.assert_frame_equal(
        in_memory_project.get("pandas_dataframe"), dataframe
    )


def test_put_pandas_series(in_memory_project):
    series = pandas.Series([0, 1, 2], index=pandas.Index([0, 1, 2], name="myIndex"))
    in_memory_project.put("pandas_series", series)
    pandas.testing.assert_series_equal(in_memory_project.get("pandas_series"), series)


def test_put_polars_dataframe(in_memory_project):
    dataframe = polars.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": [7, 8, 9],
        },
    )

    in_memory_project.put("polars_dataframe", dataframe)
    polars.testing.assert_frame_equal(
        in_memory_project.get("polars_dataframe"), dataframe
    )


def test_put_polars_series(in_memory_project):
    series = polars.Series("my_series", [0, 1, 2])
    in_memory_project.put("polars_series", series)
    polars.testing.assert_series_equal(in_memory_project.get("polars_series"), series)


def test_put_numpy_array(in_memory_project):
    # Add a Numpy array
    arr = numpy.array([1, 2, 3, 4, 5])
    in_memory_project.put("numpy_array", arr)  # NumpyArrayItem
    numpy.testing.assert_array_equal(in_memory_project.get("numpy_array"), arr)


def test_put_matplotlib_figure(in_memory_project, monkeypatch, tmp_path):
    figure, ax = subplots()
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

    in_memory_project.put("figure", figure)

    # matplotlib being not consistent (`xlink:href` are different between two calls)
    # we can't compare figures directly

    figure.savefig(tmp_path / "figure.png")
    in_memory_project.get("figure").savefig(tmp_path / "item.png")

    assert compare_images(tmp_path / "figure.png", tmp_path / "item.png", 0) is None


def test_put_altair_chart(in_memory_project):
    chart = altair.Chart().mark_point()

    in_memory_project.put("chart", chart)

    # Altair strict equality doesn't work
    assert in_memory_project.get("chart").to_json() == chart.to_json()


def test_put_pillow_image(in_memory_project):
    image1 = Image.new("RGB", (100, 100), color="red")
    image2 = Image.new("RGBA", (150, 150), color="blue")

    in_memory_project.put("image1", image1)
    in_memory_project.put("image2", image2)

    assert in_memory_project.get("image1") == image1
    assert in_memory_project.get("image2") == image2


def test_put_plotly_figure(in_memory_project):
    bar = plotly.graph_objects.Bar(x=[1, 2, 3], y=[1, 3, 2])
    figure = plotly.graph_objects.Figure(data=[bar])

    in_memory_project.put("figure", figure)

    assert in_memory_project.get("figure") == figure


def test_put_rf_model(in_memory_project, monkeypatch):
    # Add a scikit-learn model
    monkeypatch.setattr("sklearn.utils.estimator_html_repr", lambda _: "")
    model = RandomForestClassifier()
    model.fit(numpy.array([[1, 2], [3, 4]]), [0, 1])
    in_memory_project.put("rf_model", model)  # ScikitLearnModelItem
    assert isinstance(in_memory_project.get("rf_model"), RandomForestClassifier)


def test_put(in_memory_project):
    in_memory_project.put("key1", 1)
    in_memory_project.put("key2", 2)
    in_memory_project.put("key3", 3)
    in_memory_project.put("key4", 4)

    assert in_memory_project.keys() == ["key1", "key2", "key3", "key4"]


def test_put_kwargs(in_memory_project):
    in_memory_project.put(key="key1", value=1)
    assert in_memory_project.keys() == ["key1"]


def test_put_wrong_key_type(in_memory_project):
    with pytest.raises(TypeError):
        in_memory_project.put(key=2, value=1)

    assert in_memory_project.keys() == []


def test_put_twice(in_memory_project):
    in_memory_project.put("key2", 2)
    in_memory_project.put("key2", 5)

    assert in_memory_project.get("key2") == 5


def test_get(in_memory_project, mock_nowstr):
    in_memory_project.put("key", 1, note="1")
    in_memory_project.put("key", 2, note="2")

    assert in_memory_project.get("key", metadata=False) == 2
    assert in_memory_project.get("key", metadata=True) == {
        "value": 2,
        "date": mock_nowstr,
        "note": "2",
    }
    assert in_memory_project.get("key", version="all", metadata=False) == [1, 2]
    assert in_memory_project.get("key", version="all", metadata=True) == [
        {
            "value": 1,
            "date": mock_nowstr,
            "note": "1",
        },
        {
            "value": 2,
            "date": mock_nowstr,
            "note": "2",
        },
    ]
    assert in_memory_project.get("key", version=0, metadata=False) == 1
    assert in_memory_project.get("key", version=0, metadata=True) == {
        "value": 1,
        "date": mock_nowstr,
        "note": "1",
    }

    with pytest.raises(KeyError):
        in_memory_project.get("<unknown>")

    with pytest.raises(ValueError):
        in_memory_project.get("key", version=None)


def test_delete(in_memory_project):
    in_memory_project.put("key1", 1)
    in_memory_project.delete("key1")

    assert in_memory_project.keys() == []

    with pytest.raises(KeyError):
        in_memory_project.delete("key2")


def test_keys(in_memory_project):
    in_memory_project.put("key1", 1)
    in_memory_project.put("key2", 2)
    assert in_memory_project.keys() == ["key1", "key2"]


def test_put_several_complex(in_memory_project):
    in_memory_project.put("a", int)
    in_memory_project.put("b", float)

    assert in_memory_project.keys() == ["a", "b"]


def test_put_key_is_a_tuple(in_memory_project):
    """If key is not a string, warn."""
    with pytest.raises(TypeError):
        in_memory_project.put(("a", "foo"), ("b", "bar"))

    assert in_memory_project.keys() == []


def test_put_key_is_a_set(in_memory_project):
    """Cannot use an unhashable type as a key."""
    with pytest.raises(TypeError):
        in_memory_project.put(set(), "hello")

    assert in_memory_project.keys() == []


def test_put_wrong_key_and_value_raise(in_memory_project):
    """When `on_error` is "raise", raise the first error that occurs."""
    with pytest.raises(TypeError):
        in_memory_project.put(0, (lambda: "unsupported object"))


def test_put_skrub_tablereport(in_memory_project):
    from pandas import DataFrame
    from pandas.testing import assert_frame_equal
    from skrub import TableReport

    dataframe = DataFrame(dict(a=[1, 2], b=["one", "two"], c=[11.1, 11.1]))
    report = TableReport(dataframe)

    in_memory_project.put("key", report)

    assert (retrieved_report := in_memory_project.get("key"))
    assert isinstance(retrieved_report, TableReport)
    assert_frame_equal(retrieved_report.dataframe, dataframe)
