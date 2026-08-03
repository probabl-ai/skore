import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from skore import evaluate
from skore._externals._sklearn_compat import convert_container
from skore._sklearn._checks.model_checks import CheckUnderrepresentedClasses


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_detected_for_balanced_classes(report_type):
    """SKD005 does not fire when classes are balanced."""
    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=3,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=0,
    )
    report = evaluate(
        LogisticRegression(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    assert CheckUnderrepresentedClasses().check_function(report) is None


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
def test_detects_underrepresented_classes(report_type, x_container, y_container):
    """Check that the underrepresented classes issue is detected."""
    weights = [0.9, 0.05, 0.05]
    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=3,
        n_classes=len(weights),
        n_clusters_per_class=1,
        weights=weights,
        random_state=0,
    )
    feature_columns = [str(i) for i in range(X.shape[1])]
    X = convert_container(X, x_container, column_names=feature_columns)
    y = convert_container(y, y_container)
    report = evaluate(
        LogisticRegression(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    explanation = CheckUnderrepresentedClasses().check_function(report)
    assert explanation is not None
    assert "Accuracy should not be used alone" in explanation
