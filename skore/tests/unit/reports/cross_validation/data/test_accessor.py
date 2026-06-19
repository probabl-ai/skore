import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.utils._testing import _convert_container

from skore import CrossValidationReport
from skore._sklearn._plot import TableReportDisplay


@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"subsample_strategy": "invalid"}, "'subsample_strategy' options are"),
    ],
)
def test_summarize_error(regression_data, params, err_msg):
    """Check that summarize raises when subsample_strategy is invalid."""
    X, y = regression_data
    report = CrossValidationReport(LinearRegression(), X, y, splitter=2)
    with pytest.raises(ValueError, match=err_msg):
        report.data.summarize(**params)


@pytest.mark.parametrize(
    "x_container,y_container",
    [
        ("array", "array"),
        ("pandas", "series"),
        ("polars", "polars_series"),
    ],
)
def test_summarize_without_y(x_container, y_container):
    """Check summarize without target columns for several container types."""
    X, y = make_regression(n_samples=100, n_features=2, random_state=42)
    feature_columns = [f"Feature {i}" for i in range(X.shape[1])]
    X = _convert_container(
        X, x_container, column_names=feature_columns, minversion="0.20.23"
    )
    y = _convert_container(y, y_container, minversion="0.20.23")
    report = CrossValidationReport(LinearRegression(), X, y, splitter=2)

    display = report.data.summarize(with_y=False)
    assert list(display.summary["dataframe"].columns) == feature_columns


@pytest.mark.parametrize(
    "n_targets,target_column_names,x_container,y_container",
    [
        (1, ["Target"], "array", "array"),
        (1, ["Target"], "pandas", "series"),
        (1, ["Target"], "polars", "polars_series"),
        (2, ["Target 0", "Target 1"], "array", "array"),
        (2, ["Target 0", "Target 1"], "pandas", "pandas"),
        (2, ["Target 0", "Target 1"], "polars", "polars"),
    ],
)
def test_summarize_with_y(n_targets, target_column_names, x_container, y_container):
    """Check column names when summarizing with y for several container types."""
    X, y = make_regression(
        n_samples=100, n_features=2, n_targets=n_targets, random_state=42
    )
    feature_columns = [f"Feature {i}" for i in range(X.shape[1])]
    X = _convert_container(
        X, x_container, column_names=feature_columns, minversion="0.20.23"
    )
    y = _convert_container(
        y,
        y_container,
        column_names=target_column_names if n_targets > 1 else None,
        minversion="0.20.23",
    )
    report = CrossValidationReport(LinearRegression(), X, y, splitter=2)

    display = report.data.summarize(with_y=False)
    assert list(display.summary["dataframe"].columns) == feature_columns

    display = report.data.summarize(with_y=True)
    assert list(display.summary["dataframe"].columns) == (
        feature_columns + target_column_names
    )


@pytest.mark.parametrize(
    "x_container,y_container",
    [
        ("array", "array"),
        ("pandas", "series"),
        ("polars", "polars_series"),
    ],
)
@pytest.mark.parametrize("subsample_strategy", ["head", "random"])
def test_summarize_subsampling(x_container, y_container, subsample_strategy):
    """Check that subsample is handled correctly for several container types."""
    X, y = make_regression(n_samples=100, n_features=2, random_state=42)
    feature_columns = [f"Feature {i}" for i in range(X.shape[1])]
    X = _convert_container(
        X, x_container, column_names=feature_columns, minversion="0.20.23"
    )
    y = _convert_container(y, y_container, minversion="0.20.23")
    report = CrossValidationReport(LinearRegression(), X, y, splitter=2)

    display = report.data.summarize(
        subsample=10, subsample_strategy=subsample_strategy, seed=42
    )
    assert display.summary["dataframe"].shape[0] == 10

    dataframe = display.summary["dataframe"]
    if x_container != "polars" and hasattr(dataframe, "index"):
        if subsample_strategy == "head":
            assert dataframe.index.to_list() == list(range(10))
        else:
            assert dataframe.index.to_list() != list(range(10))


def test_analyze_deprecation(regression_data):
    """Check that data.analyze() emits a FutureWarning and delegates to summarize."""
    X, y = regression_data
    report = CrossValidationReport(LinearRegression(), X, y, splitter=2)

    with pytest.warns(FutureWarning, match=r"data\.analyze\(\) is deprecated"):
        display = report.data.analyze()

    assert isinstance(display, TableReportDisplay)
