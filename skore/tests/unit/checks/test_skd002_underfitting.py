import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression

from skore import EstimatorReport, evaluate
from skore._externals._sklearn_compat import convert_container
from skore._sklearn._checks._utils import CheckNotApplicable
from skore._sklearn._checks.model_checks import CheckUnderfitting


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

    if report_type == "estimator":
        report = evaluate(DummyRegressor(), X, y)
        n_metrics = report.metrics.summarize(data_source="test").summary.shape[0] - 2
    else:
        report = evaluate(DummyRegressor(), X, y, splitter=3)
        summary = report.metrics.summarize(data_source="test").summary
        n_metrics = len(
            {
                (row["verbose_name"], row["label"], row["average"], row["output"])
                for row in summary.to_dict("records")
                if row["verbose_name"] not in {"Fit time (s)", "Predict time (s)"}
            }
        )

    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD002" in issues.index
    assert (
        f"for {n_metrics}/{n_metrics} comparable metrics"
        in issues.loc["SKD002", "explanation"]
    )


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_detects_underfitting_multioutput(report_type, regression_multioutput_data):
    """SKD002 is emitted for multioutput regression when the model underfits."""
    X, y = regression_multioutput_data
    report = evaluate(
        DummyRegressor(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD002" in issues.index


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_uses_custom_metrics(report_type, binary_classification_data):
    """Check that SKD002 accounts for custom metrics added to the report."""
    X, y = binary_classification_data

    if report_type == "estimator":
        report = evaluate(DummyClassifier(), X, y, pos_label=1)
        report.metrics.add("f1")
        n_metrics = report.metrics.summarize(data_source="test").summary.shape[0] - 2
    else:
        report = evaluate(DummyClassifier(), X, y, pos_label=1, splitter=3)
        report.metrics.add("f1")
        summary = report.metrics.summarize(data_source="test").summary
        n_metrics = len(
            {
                (row["verbose_name"], row["label"], row["average"], row["output"])
                for row in summary.to_dict("records")
                if row["verbose_name"] not in {"Fit time (s)", "Predict time (s)"}
            }
        )

    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD002" in issues.index
    assert (
        f"for {n_metrics}/{n_metrics} comparable metrics"
        in issues.loc["SKD002", "explanation"]
    )


def test_not_applicable_when_train_data_missing(regression_train_test_split):
    """SKD002 needs train data to compare against a dummy baseline."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    estimator = LinearRegression().fit(X_train, y_train)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    with pytest.raises(CheckNotApplicable):
        CheckUnderfitting().check_function(report)


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_applicable_when_baseline_report_creation_fails(
    report_type, regression_data, monkeypatch
):
    """SKD002 raises when the dummy baseline report can't be fit."""
    X, y = regression_data
    report = evaluate(
        LinearRegression(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )

    def failing_fit(self, **kwargs):
        raise RuntimeError("Test error")

    monkeypatch.setattr(EstimatorReport, "_fit_estimator", failing_fit)
    with pytest.raises(CheckNotApplicable):
        CheckUnderfitting().check_function(report)
