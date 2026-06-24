import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

import skore
from skore import EstimatorReport
from skore._sklearn._plot.base import Display, DisplayMixin

DISPLAY_CLASSES = [
    getattr(skore, name)
    for name in skore.__all__
    if name.endswith("Display") and name != "Display"
]


@pytest.mark.parametrize("display_cls", DISPLAY_CLASSES)
def test_display_inherits_display_mixin(display_cls):
    assert issubclass(display_cls, DisplayMixin)


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
