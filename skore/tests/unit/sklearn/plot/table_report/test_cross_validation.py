import numpy as np
import pandas as pd
import pytest
from skore import CrossValidationReport, Display, TableReportDisplay
from skrub import tabular_pipeline
from skrub.datasets import fetch_employee_salaries


@pytest.fixture
def cross_validation_report():
    data = fetch_employee_salaries()
    X, y = data.X, data.y
    return CrossValidationReport(tabular_pipeline("regressor"), X=X, y=y)


@pytest.fixture
def display():
    data = fetch_employee_salaries()
    X, y = data.X, data.y
    report = CrossValidationReport(tabular_pipeline("regressor"), X=X, y=y)
    return report.data.analyze()


def test_table_report_display_constructor(display):
    """Check the value that are stored in the display constructor."""
    assert isinstance(display, Display)

    assert hasattr(display, "summary")
    assert isinstance(display.summary, dict)
    assert list(display.summary.keys()) == [
        "dataframe",
        "dataframe_module",
        "n_rows",
        "n_columns",
        "columns",
        "dataframe_is_empty",
        "plots_skipped",
        "associations_skipped",
        "sample_table",
        "n_constant_columns",
        "top_associations",
    ]


def test_table_report_display_frame(cross_validation_report):
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
        # pd.Series(np.ones(100)),
        # pd.DataFrame(np.ones((100, 1)), columns=["Target"]),
    ],
)
def test_retrieve_data_as_frame(X, y):
    report = CrossValidationReport(tabular_pipeline("regressor"), X=X, y=y)
    display = report.data.analyze()
    assert isinstance(display, TableReportDisplay)


def test_without_y(cross_validation_report):
    """Check that the data accessor works without y."""
    display = cross_validation_report.data.analyze(with_y=False)
    assert isinstance(display, TableReportDisplay)

    df = display.frame(kind="dataset")
    assert "gender" in df.columns
    assert "current_annual_salary" not in df.columns
