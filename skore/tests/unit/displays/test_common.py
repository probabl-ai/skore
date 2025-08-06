import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from skore import EstimatorReport
from skore._sklearn._plot.base import Display


@pytest.mark.parametrize(
    "plot_func, estimator, dataset",
    [
        ("roc", LogisticRegression(), make_classification(random_state=42)),
        (
            "precision_recall",
            LogisticRegression(),
            make_classification(random_state=42),
        ),
        ("prediction_error", LinearRegression(), make_regression(random_state=42)),
    ],
)
def test_display_help(pyplot, capsys, plot_func, estimator, dataset):
    """Check that the help method writes to the console."""

    X_train, X_test, y_train, y_test = train_test_split(*dataset, random_state=42)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = getattr(report.metrics, plot_func)()

    display.help()
    captured = capsys.readouterr()
    assert f"{display.__class__.__name__}" in captured.out


@pytest.mark.parametrize(
    "plot_func, estimator, dataset",
    [
        ("roc", LogisticRegression(), make_classification(random_state=42)),
        (
            "precision_recall",
            LogisticRegression(),
            make_classification(random_state=42),
        ),
        ("prediction_error", LinearRegression(), make_regression(random_state=42)),
    ],
)
def test_display_str(pyplot, plot_func, estimator, dataset):
    """Check that __str__ returns a string starting with the expected prefix."""
    X_train, X_test, y_train, y_test = train_test_split(*dataset, random_state=42)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = getattr(report.metrics, plot_func)()

    str_str = str(display)
    assert f"{display.__class__.__name__}" in str_str
    assert "help()" in str_str


@pytest.mark.parametrize(
    "plot_func, estimator, dataset",
    [
        ("roc", LogisticRegression(), make_classification(random_state=42)),
        (
            "precision_recall",
            LogisticRegression(),
            make_classification(random_state=42),
        ),
        ("prediction_error", LinearRegression(), make_regression(random_state=42)),
    ],
)
def test_display_repr(pyplot, plot_func, estimator, dataset):
    """Check that __repr__ returns a string starting with the expected prefix."""
    X_train, X_test, y_train, y_test = train_test_split(*dataset, random_state=42)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = getattr(report.metrics, plot_func)()

    repr_str = repr(display)
    assert f"skore.{display.__class__.__name__}(...)" in repr_str


@pytest.mark.parametrize(
    "plot_func, estimator, dataset",
    [
        ("roc", LogisticRegression(), make_classification(random_state=42)),
        (
            "precision_recall",
            LogisticRegression(),
            make_classification(random_state=42),
        ),
        ("prediction_error", LinearRegression(), make_regression(random_state=42)),
    ],
)
def test_display_protocol(pyplot, capsys, plot_func, estimator, dataset):
    """Check that the display object adheres to the Display protocol."""

    X_train, X_test, y_train, y_test = train_test_split(*dataset, random_state=42)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = getattr(report.metrics, plot_func)()

    assert isinstance(display, Display)
