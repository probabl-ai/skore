import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression

from skore import EstimatorReport, evaluate
from skore._externals._sklearn_compat import convert_container
from skore._sklearn._checks._utils import CheckNotApplicable
from skore._sklearn._checks.model_checks import CheckUnderfitting


@pytest.mark.parametrize(
    "x_container, y_container",
    [
        ("array", "array"),
        ("pandas", "series"),
        ("polars", "polars_series"),
    ],
)
@pytest.mark.filterwarnings("ignore:X does not have valid feature names:UserWarning")
def test_detects_underfitting(regression_data, x_container, y_container):
    """Check that the underfitting issue is detected."""
    X, y = regression_data
    feature_columns = [str(i) for i in range(X.shape[1])]
    X = convert_container(X, x_container, column_names=feature_columns)
    y = convert_container(y, y_container)
    report = evaluate(DummyRegressor(), X, y)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    n_metrics = report.metrics.summarize(data_source="test").summary.shape[0] - 2
    assert "SKD002" in issues.index
    assert (
        f"for {n_metrics}/{n_metrics} comparable metrics"
        in issues.loc["SKD002", "explanation"]
    )


@pytest.mark.parametrize(
    "x_container,y_container",
    [
        ("array", "array"),
        ("pandas", "series"),
        ("polars", "polars_series"),
    ],
)
@pytest.mark.filterwarnings("ignore:X does not have valid feature names:UserWarning")
def test_detects_underfitting_cv(regression_data, x_container, y_container):
    """Check that the underfitting issue is detected on a cross-validation report."""
    X, y = regression_data
    feature_columns = [str(i) for i in range(X.shape[1])]
    X = convert_container(X, x_container, column_names=feature_columns)
    y = convert_container(y, y_container)
    report = evaluate(DummyRegressor(), X, y, splitter=3)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    summary = report.metrics.summarize(data_source="test").summary
    n_metrics = len(
        {
            (row["verbose_name"], row["label"], row["average"], row["output"])
            for row in summary.to_dict("records")
            if row["verbose_name"] not in {"Fit time (s)", "Predict time (s)"}
        }
    )
    assert "SKD002" in issues.index
    assert (
        f"for {n_metrics}/{n_metrics} comparable metrics"
        in issues.loc["SKD002", "explanation"]
    )


def test_detects_underfitting_multioutput(regression_multioutput_data):
    """SKD002 is emitted for multioutput regression when the model underfits."""
    X, y = regression_multioutput_data
    report = evaluate(DummyRegressor(), X, y)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD002" in issues.index


def test_uses_custom_metrics(binary_classification_data):
    """Check that SKD002 accounts for custom metrics added to the report."""
    X, y = binary_classification_data
    report = evaluate(DummyClassifier(), X, y, pos_label=1)
    report.metrics.add("f1")
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    n_metrics = report.metrics.summarize(data_source="test").summary.shape[0] - 2
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


def test_not_applicable_when_baseline_report_creation_fails(
    regression_data, monkeypatch
):
    """SKD002 raises when the dummy baseline report can't be fit."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)

    def failing_fit(self, **kwargs):
        raise RuntimeError("Test error")

    monkeypatch.setattr(EstimatorReport, "_fit_estimator", failing_fit)
    with pytest.raises(CheckNotApplicable):
        CheckUnderfitting().check_function(report)


def test_not_applicable_when_baseline_report_creation_fails_cv(
    regression_data, monkeypatch
):
    """SKD002 raises on a CV report when the dummy baseline report can't be fit."""
    X, y = regression_data
    cv_report = evaluate(LinearRegression(), X, y, splitter=3)

    def failing_fit(self, **kwargs):
        raise RuntimeError("Test error")

    monkeypatch.setattr(EstimatorReport, "_fit_estimator", failing_fit)
    with pytest.raises(CheckNotApplicable):
        CheckUnderfitting().check_function(cv_report)
