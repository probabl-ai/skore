import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from skore import EstimatorReport, evaluate
from skore._sklearn._checks._utils import CheckNotApplicable
from skore._sklearn._checks.model_checks import CheckOverfitting


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_detects_overfitting(report_type, regression_data):
    """If overfitting happens, the overfitting check fires."""
    X, y = regression_data

    if report_type == "estimator":
        report = evaluate(DecisionTreeRegressor(random_state=0), X, y)
        n_metrics = report.metrics.summarize(data_source="test").summary.shape[0] - 2
    else:
        report = evaluate(DecisionTreeRegressor(random_state=0), X, y, splitter=3)
        summary = report.metrics.summarize(data_source="test").summary
        n_metrics = len(
            {
                (row["verbose_name"], row["label"], row["average"], row["output"])
                for row in summary.to_dict("records")
                if row["verbose_name"] not in {"Fit time (s)", "Predict time (s)"}
            }
        )

    explanation = CheckOverfitting().check_function(report)
    assert explanation is not None
    assert f"for {n_metrics}/{n_metrics} default predictive metrics" in explanation


def test_not_applicable_when_train_data_missing(regression_train_test_split):
    """SKD001 needs train data to compare against test scores."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    estimator = LinearRegression().fit(X_train, y_train)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    with pytest.raises(CheckNotApplicable, match="Train data is unavailable."):
        CheckOverfitting().check_function(report)
