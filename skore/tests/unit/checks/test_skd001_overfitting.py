import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from skore import EstimatorReport, evaluate
from skore._checks.model_checks import CheckOverfitting
from skore._checks.utils import CheckNotApplicable


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_detects_overfitting(report_type, regression_data):
    """If overfitting happens, the overfitting check fires."""
    X, y = regression_data
    report = evaluate(
        DecisionTreeRegressor(random_state=0),
        X,
        y,
        splitter=0.2 if report_type == "estimator" else 3,
    )
    n_metrics = (
        report.metrics.summarize(data_source="test")
        .frame(aggregate="mean", flat_index=True)
        .shape[0]
        - 2
    )

    explanation = CheckOverfitting().check_function(report)
    assert explanation is not None
    assert f"for {n_metrics}/{n_metrics} default predictive metrics" in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_detects_overfitting_multioutput(report_type, regression_multioutput_data):
    """SKD001 is emitted for multioutput regression when the model overfits."""
    X, y = regression_multioutput_data
    report = evaluate(
        DecisionTreeRegressor(random_state=0),
        X,
        y,
        splitter=0.2 if report_type == "estimator" else 3,
    )
    assert CheckOverfitting().check_function(report) is not None


def test_not_applicable_when_train_data_missing(regression_train_test_split):
    """SKD001 needs train data to compare against test scores."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    estimator = LinearRegression().fit(X_train, y_train)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    with pytest.raises(CheckNotApplicable, match="Train data is unavailable."):
        CheckOverfitting().check_function(report)


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_uses_custom_metrics(report_type, regression_data):
    """SKD001 accounts for custom metrics added to the report."""
    X, y = regression_data
    report = evaluate(
        DecisionTreeRegressor(random_state=0),
        X,
        y,
        splitter=0.2 if report_type == "estimator" else 3,
    )
    report.metrics.add("root_mean_squared_error")
    n_metrics = len(
        [
            m
            for m in report.metrics.available()
            if m not in ["score", "fit_time", "predict_time"]
        ]
    )

    explanation = CheckOverfitting().check_function(report)
    assert explanation is not None
    assert f"for {n_metrics}/{n_metrics} default predictive metrics" in explanation
