import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from skore import evaluate
from skore._checks.skd004_high_class_imbalance import CheckHighClassImbalance
from skore._externals.sklearn_compat import convert_container


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_detected_for_balanced_classes(report_type):
    """SKD004 does not fire when classes are balanced."""
    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=3,
        n_classes=2,
        random_state=0,
    )
    report = evaluate(
        LogisticRegression(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    assert CheckHighClassImbalance().check_function(report) is None


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
@pytest.mark.parametrize(
    "x_container,y_container",
    [
        ("array", "array"),
        ("pandas", "series"),
        ("polars", "polars_series"),
    ],
)
@pytest.mark.filterwarnings("ignore:X does not have valid feature names:UserWarning")
def test_detects_high_class_imbalance(report_type, x_container, y_container):
    """Check that the high class imbalance issue is detected."""
    weights = [0.9, 0.1]
    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=3,
        n_classes=len(weights),
        weights=weights,
        random_state=0,
    )
    feature_columns = [str(i) for i in range(X.shape[1])]
    X = convert_container(X, x_container, column_names=feature_columns)
    y = convert_container(y, y_container)
    report = evaluate(
        LogisticRegression(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    explanation = CheckHighClassImbalance().check_function(report)
    assert explanation is not None
    assert "Accuracy should not be used alone" in explanation
