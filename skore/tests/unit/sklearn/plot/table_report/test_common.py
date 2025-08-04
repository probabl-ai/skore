import pandas as pd
import pytest
from matplotlib.collections import QuadMesh
from sklearn.model_selection import train_test_split
from skore import EstimatorReport
from skore._sklearn._plot.data.table_report import (
    _truncate_top_k_categories,
)
from skrub import tabular_learner
from skrub.datasets import fetch_employee_salaries


@pytest.fixture
def display():
    data = fetch_employee_salaries()
    X, y = data.X, data.y
    X["gender"] = X["gender"].astype("category")
    X["date_first_hired"] = pd.to_datetime(X["date_first_hired"])
    X["timedelta_hired"] = (
        pd.Timestamp.now() - X["date_first_hired"]
    ).dt.to_pytimedelta()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    report = EstimatorReport(
        tabular_learner("regressor"),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    return report.data.analyze()


@pytest.fixture
def estimator_report():
    data = fetch_employee_salaries()
    X, y = data.X, data.y
    X["gender"] = X["gender"].astype("category")
    X["date_first_hired"] = pd.to_datetime(X["date_first_hired"])
    X["cents"] = 100 * y
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return EstimatorReport(
        tabular_learner("regressor"),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


@pytest.mark.parametrize("col", [None, pd.Series(range(10))])
def test_truncate_top_k_categories_return_as_is(col):
    """Check the behaviour of `_truncate_top_k_categories` when `col` is None or
    numeric where no changes are made."""
    assert _truncate_top_k_categories(col, k=3) is col


def test_corr_plot(pyplot, estimator_report):
    display = estimator_report.data.analyze(data_source="train")
    display.plot(kind="corr")
    assert isinstance(display.ax_.collections[0], QuadMesh)
    assert len(display.ax_.get_xticklabels()) == 10
    assert len(display.ax_.get_yticklabels()) == 10
    assert display.ax_.title.get_text() == "Cramer's V Correlation"


def test_repr(display):
    repr = display.__repr__()
    assert repr == "<TableReportDisplay(...)>"
