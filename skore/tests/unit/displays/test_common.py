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
def test_display_protocol(pyplot, capsys, plot_func, estimator, dataset):
    """Check that the display object adheres to the Display protocol."""

    X_train, X_test, y_train, y_test = train_test_split(*dataset, random_state=42)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = getattr(report.metrics, plot_func)()

    assert isinstance(display, Display)


def test_repeated_display_plot_calls_do_not_keep_figures_open(pyplot, tmp_path):
    """Repeated plotting should not accumulate pyplot-managed figures."""

    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    report = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.metrics.precision_recall()

    for idx in range(25):
        fig = display.plot()
        fig.savefig(tmp_path / f"precision-recall-{idx}.svg")

    assert pyplot.get_fignums() == []
