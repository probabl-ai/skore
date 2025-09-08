import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_regression
from skore import CrossValidationReport, Display, TableReportDisplay
from skore._externals._skrub_compat import tabular_pipeline


@pytest.fixture
def cross_validation_report():
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(5)])
    y = pd.Series(y, name="Target_")
    return CrossValidationReport(tabular_pipeline("regressor"), X=X, y=y)


@pytest.fixture
def display(cross_validation_report):
    return cross_validation_report.data.analyze()


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
            # skrub>=0.6
            # "associations_skipped",
            "sample_table",
            "n_constant_columns",
            "top_associations",
        }
    )


def test_table_report_display_frame(cross_validation_report):
    """Check that we return the expected kind of data when calling `.frame`."""
    display = cross_validation_report.data.analyze()
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
    "X",
    [
        np.random.rand(100, 5),
        pd.DataFrame(
            np.random.rand(100, 5), columns=[f"Feature number {i}" for i in range(5)]
        ),
    ],
)
@pytest.mark.parametrize(
    "y",
    [
        np.ones((100, 1)),
        np.ones(100),
        pd.Series(np.ones(100)),
        pd.DataFrame(np.ones((100, 1)), columns=["Target"]),
    ],
)
def test_display_creation(X, y):
    """Check that the display can be created with different types of X and y."""
    report = CrossValidationReport(tabular_pipeline("regressor"), X=X, y=y)
    display = report.data.analyze()
    assert isinstance(display, TableReportDisplay)


def test_without_y(cross_validation_report):
    """Check that the data accessor works without y."""
    display = cross_validation_report.data.analyze(with_y=False)
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
def test_analyze_with_subsample(
    cross_validation_report, sumbsample, subsample_strategy, seed
):
    """Check that the analyze method works with subsampling."""
    display = cross_validation_report.data.analyze(
        subsample=sumbsample,
        subsample_strategy=subsample_strategy,
        seed=seed,
    )
    assert isinstance(display, TableReportDisplay)
    assert len(display.frame(kind="dataset")) == sumbsample


def test_analyze_with_invalid_subsample_strategy(cross_validation_report):
    """Check that an error is raised with an invalid subsample strategy."""
    with pytest.raises(ValueError):
        cross_validation_report.data.analyze(
            subsample=10,
            subsample_strategy="invalid_strategy",
        )


def test_check_y_required():
    """Check that we raise an error when y is not provided."""
    X = np.random.rand(100, 5)
    with pytest.raises(ValueError):
        CrossValidationReport(KMeans(), X=X).data.analyze()


def test_repr(cross_validation_report):
    """Check that __repr__ returns a string starting with the expected prefix."""
    repr_str = repr(cross_validation_report.data)
    assert "CrossValidationReport" in repr_str
