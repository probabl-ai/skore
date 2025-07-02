import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from skore import Display, EstimatorReport
from skrub import tabular_learner
from skrub.datasets import fetch_employee_salaries


@pytest.fixture
def skrub_data():
    data = fetch_employee_salaries()
    return train_test_split(data.X, data.y, random_state=0)


@pytest.fixture
def estimator_report():
    data = fetch_employee_salaries()
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, random_state=0)
    return EstimatorReport(
        tabular_learner("regressor"),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def test_table_report_display_constructor(estimator_report):
    """Check the value that are stored in the display constructor."""
    display = estimator_report.data.analyze()
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
        "sample_table",
        "n_constant_columns",
        "top_associations",
    ]


def test_table_report_display_frame_error(estimator_report):
    """Check the error message when passing an unknown kind in the `frame` method."""
    display = estimator_report.data.analyze()
    with pytest.raises(ValueError, match="Invalid kind: 'xxx'"):
        display.frame(kind="xxx")


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_table_report_display_frame(estimator_report, data_source):
    display = estimator_report.data.analyze(data_source=data_source)
    dataset = display.frame(kind="dataset")

    if data_source == "train":
        pd.testing.assert_frame_equal(
            dataset,
            pd.concat([estimator_report.X_train, estimator_report.y_train], axis=1),
        )
    else:
        pd.testing.assert_frame_equal(
            dataset,
            pd.concat([estimator_report.X_test, estimator_report.y_test], axis=1),
        )

    associations = display.frame(kind="top-associations")
    pd.testing.assert_frame_equal(
        associations, pd.DataFrame(display.summary["top_associations"])
    )
