import pytest
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV

from skore import EstimatorReport, evaluate
from skore._sklearn._checks._utils import CheckNotApplicable
from skore._sklearn._checks.model_checks import CheckWorseThanBaseline


def test_detects_worse_than_baseline(regression_data):
    """Check that the worse-than-baseline issue is detected on a dummy estimator."""
    X, y = regression_data
    report = evaluate(DummyRegressor(), X, y)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD009" in issues.index
    assert (
        "not significantly better than a HistGradientBoosting baseline"
        in issues.loc["SKD009", "explanation"]
    )


@pytest.mark.filterwarnings(
    "ignore:Only pandas and polars DataFrames are supported:UserWarning:skrub"
)
def test_detects_worse_than_baseline_cv(regression_data):
    """Check that the worse-than-baseline issue is detected on a CV report."""
    X, y = regression_data
    report = evaluate(DummyRegressor(), X, y, splitter=3)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD009" in issues.index
    assert (
        "not significantly better than a HistGradientBoosting baseline"
        in issues.loc["SKD009", "explanation"]
    )


def test_not_detected_on_strong_model(regression_data):
    """Check that SKD009 is not detected when the model beats HistGradientBoosting."""
    X, y = make_regression(n_features=4, noise=0.1, random_state=0)
    report = evaluate(RidgeCV(), X, y)
    codes = set(report.checks.summarize().frame(section="issue")["code"])
    assert "SKD009" not in codes


def test_not_detected_on_strong_model_cv():
    """Check that SKD009 is not detected on a CV report for a strong model."""
    X, y = make_regression(n_features=4, noise=0.1, random_state=0)
    report = evaluate(RidgeCV(), X, y, splitter=3)
    codes = set(report.checks.summarize().frame(section="issue")["code"])
    assert "SKD009" not in codes


def test_detects_worse_than_baseline_multioutput(regression_multioutput_data):
    """SKD009 emitted for multioutput regression when model is worse than baseline."""
    X, y = regression_multioutput_data
    report = evaluate(DummyRegressor(), X, y)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD009" in issues.index


@pytest.mark.filterwarnings(
    "ignore:Only pandas and polars DataFrames are supported:UserWarning:skrub"
)
def test_detects_worse_than_baseline_multioutput_cv(
    regression_multioutput_data,
):
    """SKD009 emitted for multioutput regression on a CV report."""
    X, y = regression_multioutput_data
    report = evaluate(DummyRegressor(), X, y, splitter=3)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD009" in issues.index


def test_not_applicable_when_train_data_missing(regression_train_test_split):
    """SKD009 needs train data to build the baseline comparison."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    estimator = LinearRegression().fit(X_train, y_train)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    with pytest.raises(CheckNotApplicable):
        CheckWorseThanBaseline().check_function(report)


def test_not_applicable_when_baseline_report_creation_fails(
    regression_data, monkeypatch
):
    """SKD009 raises when the HistGradientBoosting baseline report can't be fit."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)

    def failing_fit(self, **kwargs):
        raise RuntimeError("Test error")

    monkeypatch.setattr(EstimatorReport, "_fit_estimator", failing_fit)
    with pytest.raises(CheckNotApplicable):
        CheckWorseThanBaseline().check_function(report)


def test_not_applicable_when_baseline_report_creation_fails_cv(
    regression_data, monkeypatch
):
    """SKD009 raises on a CV report when the baseline report can't be fit."""
    X, y = regression_data
    cv_report = evaluate(LinearRegression(), X, y, splitter=3)

    def failing_fit(self, **kwargs):
        raise RuntimeError("Test error")

    monkeypatch.setattr(EstimatorReport, "_fit_estimator", failing_fit)
    with pytest.raises(CheckNotApplicable):
        CheckWorseThanBaseline().check_function(cv_report)
