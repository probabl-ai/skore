import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from skore import evaluate
from skore._checks.skd002_underfitting import CheckUnderfitting
from skore._externals.sklearn_compat import convert_container


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
@pytest.mark.parametrize(
    "x_container, y_container",
    [
        ("array", "array"),
        ("pandas", "series"),
        ("polars", "polars_series"),
    ],
)
@pytest.mark.filterwarnings("ignore:X does not have valid feature names:UserWarning")
def test_detects_underfitting(report_type, regression_data, x_container, y_container):
    """Check that the underfitting issue is detected."""
    X, y = regression_data
    feature_columns = [str(i) for i in range(X.shape[1])]
    X = convert_container(X, x_container, column_names=feature_columns)
    y = convert_container(y, y_container)
    report = evaluate(
        DummyRegressor(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    n_metrics = (
        report.metrics.summarize(data_source="test")
        .frame(aggregate="mean", flat_index=True)
        .shape[0]
        - 2
    )

    explanation = CheckUnderfitting().check_function(report)
    assert explanation is not None
    assert f"for {n_metrics}/{n_metrics} comparable metrics" in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_detects_underfitting_multioutput(report_type, regression_multioutput_data):
    """SKD002 is emitted for multioutput regression when the model underfits."""
    X, y = regression_multioutput_data
    report = evaluate(
        DummyRegressor(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    assert CheckUnderfitting().check_function(report) is not None


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_uses_custom_metrics(report_type, binary_classification_data):
    """Check that SKD002 accounts for custom metrics added to the report."""
    X, y = binary_classification_data

    report = evaluate(
        DummyClassifier(),
        X,
        y,
        pos_label=1,
        splitter=0.2 if report_type == "estimator" else 3,
    )
    report.metrics.add("f1")
    # DummyClassifier's score() method is not exactly `ClassifierMixin`'s so
    # it is considered as a proper metric and used in the check
    n_metrics = len(
        [m for m in report.metrics.available() if m not in ["fit_time", "predict_time"]]
    )

    explanation = CheckUnderfitting().check_function(report)
    assert explanation is not None
    assert f"for {n_metrics}/{n_metrics} comparable metrics" in explanation
