import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from skrub import tabular_pipeline

from skore import CrossValidationReport, Display, TableReportDisplay
from skore._externals._sklearn_compat import convert_container


@pytest.fixture(scope="module")
def cross_validation_report():
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(5)])
    y = pd.Series(y, name="Target_")
    return CrossValidationReport(tabular_pipeline(DummyRegressor()), X=X, y=y)


@pytest.fixture(scope="module")
def display(cross_validation_report):
    return cross_validation_report.data.summarize()


def test_table_report_display_constructor(display):
    """Check the value that are stored in the display constructor."""
    assert isinstance(display, Display)

    assert hasattr(display, "summary")
    assert isinstance(display.summary, dict)
    # checking that we have all the keys required to create a TableReportDisplay
    assert set(display.summary.keys()).issuperset(
        {
            "dataframe",
            "dataframe_module",
            "n_rows",
            "n_columns",
            "columns",
            "dataframe_is_empty",
            "plots_skipped",
            "sample_table",
            "n_constant_columns",
            "top_associations",
        }
    )


def test_table_report_display_frame(cross_validation_report, display):
    """Check that we return the expected kind of data when calling `.frame`."""
    dataset = display.frame(kind="dataset")

    pd.testing.assert_frame_equal(
        dataset,
        pd.concat([cross_validation_report.X, cross_validation_report.y], axis=1),
    )

    associations = display.frame(kind="top-associations")
    pd.testing.assert_frame_equal(
        associations, pd.DataFrame(display.summary["top_associations"])
    )


@pytest.mark.parametrize(
    "x_container,y_container",
    [
        ("array", "array"),
        ("pandas", "series"),
        ("polars", "polars_series"),
    ],
)
def test_display_creation_with_containers(x_container, y_container):
    """Check that the display can be created with paired container types."""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    feature_columns = [f"Feature_{i}" for i in range(X.shape[1])]
    X = convert_container(
        X, x_container, column_names=feature_columns, minversion="0.20.23"
    )
    y = convert_container(y, y_container, minversion="0.20.23")
    report = CrossValidationReport(tabular_pipeline(DummyRegressor()), X=X, y=y)
    display = report.data.summarize()
    assert isinstance(display, TableReportDisplay)


@pytest.mark.parametrize(
    "X",
    [
        np.random.rand(100, 5),
        pd.DataFrame(np.random.rand(100, 5)),
        pd.DataFrame(
            np.random.rand(100, 5), columns=[f"Feature number {i}" for i in range(5)]
        ),
        pd.DataFrame(np.random.rand(100, 5), columns=["a", 1, "c", 3, "e"]),
    ],
)
@pytest.mark.parametrize(
    "y",
    [
        np.ones((100, 1)),
        np.ones(100),
        pd.Series(np.ones(100)),
        pd.Series(np.ones(100), name="Target"),
        pd.DataFrame(np.ones((100, 1))),
        pd.DataFrame(np.ones((100, 1)), columns=["Target"]),
    ],
)
def test_display_creation(X, y):
    """Check that the display can be created with different types of X and y."""
    report = CrossValidationReport(tabular_pipeline(DummyRegressor()), X=X, y=y)
    display = report.data.summarize()
    assert isinstance(display, TableReportDisplay)


def test_without_y(cross_validation_report):
    """Check that the data accessor works without y."""
    display = cross_validation_report.data.summarize(with_y=False)
    assert isinstance(display, TableReportDisplay)

    df = display.frame(kind="dataset")
    assert "Feature_0" in df.columns
    assert "Target_" not in df.columns


@pytest.mark.parametrize(
    "sumbsample, subsample_strategy, seed",
    [
        (10, "head", None),
        (10, "random", 42),
    ],
)
def test_summarize_with_subsample(
    cross_validation_report, sumbsample, subsample_strategy, seed
):
    """Check that the summarize method works with subsampling."""
    display = cross_validation_report.data.summarize(
        subsample=sumbsample,
        subsample_strategy=subsample_strategy,
        seed=seed,
    )
    assert isinstance(display, TableReportDisplay)
    assert len(display.frame(kind="dataset")) == sumbsample


def test_summarize_with_invalid_subsample_strategy(cross_validation_report):
    """Check that an error is raised with an invalid subsample strategy."""
    with pytest.raises(ValueError):
        cross_validation_report.data.summarize(
            subsample=10,
            subsample_strategy="invalid_strategy",
        )


def test_analyze_deprecation(cross_validation_report):
    """Check that data.analyze() emits a FutureWarning and delegates to summarize."""
    with pytest.warns(FutureWarning, match=r"data\.analyze\(\) is deprecated"):
        display = cross_validation_report.data.analyze()

    assert isinstance(display, TableReportDisplay)
