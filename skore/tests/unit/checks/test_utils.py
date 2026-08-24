from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skore import CrossValidationReport, EstimatorReport
from skore._checks.utils import (
    CheckNotApplicable,
    adaptive_threshold,
    cast_report,
    check_score_better_than_baseline,
    collect_scores,
    get_fitted_estimator,
    get_preprocessed_X,
    get_report_y,
    majority_vote,
    split_preprocessor_estimator,
)
from skore._checks.utils import baseline_estimator_report as _baseline_estimator_report
from skore._utils.testing import MockReport


@pytest.fixture
def small_estimator_report(regression_data):
    X, y = regression_data
    return EstimatorReport(
        LinearRegression(), X_train=X[:60], y_train=y[:60], X_test=X[60:], y_test=y[60:]
    )


@pytest.fixture
def small_cv_report(regression_data):
    X, y = regression_data
    return CrossValidationReport(LinearRegression(), X=X, y=y, splitter=3)


# adaptive_threshold


def test_adaptive_threshold_uses_floor_when_larger():
    """The floor wins when the scaled fraction is smaller."""
    assert adaptive_threshold(floor=0.1, fraction=0.05, references=(1.0,)) == 0.1


def test_adaptive_threshold_scales_with_reference_magnitude():
    """The fraction of the largest-magnitude reference wins when above the floor."""
    assert adaptive_threshold(floor=0.01, fraction=0.1, references=(100.0,)) == 10.0


def test_adaptive_threshold_uses_largest_absolute_reference():
    """Multiple references: the largest absolute value drives the threshold."""
    assert adaptive_threshold(floor=0.0, fraction=0.5, references=(-10.0, 2.0)) == 5.0


# check_score_better_than_baseline


@pytest.mark.parametrize(
    "greater_is_better, score, baseline, expected",
    [
        (True, 1.0, 0.8, True),  # gap of 0.2 exceeds floor/fraction
        (True, 0.81, 0.8, False),  # gap too small
        (False, 0.8, 1.0, True),  # lower is better, gap of 0.2
        (False, 0.99, 1.0, False),  # gap too small
        (True, 0.13, 0.1, True),  # gap of 0.03 hits the floor (fraction*baseline=0.01)
    ],
)
def test_check_score_better_than_baseline(greater_is_better, score, baseline, expected):
    """The gap direction follows `greater_is_better` and is floored/scaled."""
    assert (
        check_score_better_than_baseline(
            score, baseline, greater_is_better, floor=0.03, fraction=0.1
        )
        == expected
    )


def test_check_score_better_than_baseline_nan_greater_is_better_returns_false():
    """A NaN `greater_is_better` (e.g. an undefined metric) never signals a gap."""
    assert (
        check_score_better_than_baseline(
            1.0, 0.0, float("nan"), floor=0.0, fraction=0.0
        )
        is False
    )


# majority vote


@pytest.mark.parametrize(
    "votes, expected",
    [
        ([True, True, False], (True, 2, 3)),
        ([True, False, False], (False, 1, 3)),
        ([True, False], (False, 1, 2)),  # exactly half is not a majority
        ([], (False, 0, 0)),
    ],
)
def test_majority_vote(votes, expected):
    assert majority_vote(votes) == expected


# split_preprocessor_estimator


def test_split_preprocessor_estimator_plain_estimator():
    preprocessor, predictor = split_preprocessor_estimator(LinearRegression())
    assert preprocessor is None
    assert isinstance(predictor, LinearRegression)


def test_split_preprocessor_estimator_single_step_pipeline():
    """A single-step pipeline has no preprocessor, only the predictor."""
    pipeline = Pipeline([("model", LinearRegression())])
    preprocessor, predictor = split_preprocessor_estimator(pipeline)
    assert preprocessor is None
    assert isinstance(predictor, LinearRegression)


def test_split_preprocessor_estimator_multi_step_pipeline():
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    preprocessor, predictor = split_preprocessor_estimator(pipeline)
    assert isinstance(preprocessor, Pipeline)
    assert list(preprocessor.named_steps) == ["scaler"]
    assert isinstance(predictor, LinearRegression)


# cast_report


def test_cast_report_is_a_no_op_passthrough(small_estimator_report, small_cv_report):
    """`cast_report` only narrows the static type; the object is unchanged."""
    assert cast_report(small_estimator_report) is small_estimator_report
    assert cast_report(small_cv_report) is small_cv_report


# get_fitted_estimator


def test_get_fitted_estimator_estimator_report(small_estimator_report):
    assert get_fitted_estimator(small_estimator_report) is (
        small_estimator_report.estimator_
    )


def test_get_fitted_estimator_cv_report_uses_first_split(small_cv_report):
    assert get_fitted_estimator(small_cv_report) is (
        small_cv_report.reports_[0].estimator_
    )


# get_report_y


def test_get_report_y_estimator_both_concatenates_train_and_test(
    small_estimator_report,
):
    y = get_report_y(small_estimator_report, data_source="both")
    assert len(y) == len(small_estimator_report.y_train) + len(
        small_estimator_report.y_test
    )


def test_get_report_y_estimator_train_only(small_estimator_report):
    y = get_report_y(small_estimator_report, data_source="train")
    assert len(y) == len(small_estimator_report.y_train)


def test_get_report_y_cv_report_ignores_data_source(small_cv_report):
    """For CV reports the full dataset target is returned regardless of data_source."""
    y_both = get_report_y(small_cv_report, data_source="both")
    y_train = get_report_y(small_cv_report, data_source="train")
    assert len(y_both) == len(y_train) == len(small_cv_report.y)


@pytest.mark.parametrize("data_source", ["both", "train"])
def test_get_report_y_raises_not_applicable_when_train_missing(
    regression_data, data_source
):
    X, y = regression_data
    fitted = LinearRegression().fit(X, y)
    report = EstimatorReport(fitted, X_test=X, y_test=y)
    with pytest.raises(CheckNotApplicable, match="Target train data is unavailable"):
        get_report_y(report, data_source=data_source)


def test_get_report_y_raises_not_applicable_for_sparse_target():
    """A sparse target cannot be normalized into a dense Series/DataFrame."""
    X = np.random.default_rng(0).normal(size=(10, 2))
    y_sparse = sp.random(10, 1, density=0.5, format="csr")
    report = MockReport(
        LinearRegression().fit(X, np.random.default_rng(0).normal(size=10)),
        X_train=X,
        y_train=y_sparse,
        X_test=X,
        y_test=np.random.default_rng(0).normal(size=10),
    )
    report._report_type = "estimator"
    with pytest.raises(CheckNotApplicable, match="Target data is sparse"):
        get_report_y(report, data_source="train")


# get_preprocessed_X


def test_get_preprocessed_X_plain_estimator_returns_raw_features(
    small_estimator_report,
):
    X = get_preprocessed_X(small_estimator_report, data_source="test")
    assert len(X) == len(small_estimator_report.X_test)


def test_get_preprocessed_X_applies_pipeline_preprocessor(regression_data):
    """A pipeline estimator's features are passed through its preprocessor."""
    X, y = regression_data
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    report = EstimatorReport(
        pipeline, X_train=X[:60], y_train=y[:60], X_test=X[60:], y_test=y[60:]
    )
    preprocessed = get_preprocessed_X(report, data_source="train")
    raw = pd.DataFrame(report.X_train)
    # A fitted StandardScaler centers each column to (approximately) zero mean.
    assert np.allclose(np.asarray(preprocessed).mean(axis=0), 0, atol=1e-8)
    assert not np.allclose(raw.mean(axis=0), 0, atol=1e-8)


def test_get_preprocessed_X_cv_report_ignores_data_source(small_cv_report):
    X_both = get_preprocessed_X(small_cv_report, data_source="both")
    X_train = get_preprocessed_X(small_cv_report, data_source="train")
    assert len(X_both) == len(X_train) == len(small_cv_report.X)


def test_get_preprocessed_X_raises_not_applicable_when_train_missing(regression_data):
    X, y = regression_data
    fitted = LinearRegression().fit(X, y)
    report = EstimatorReport(fitted, X_test=X, y_test=y)
    with pytest.raises(CheckNotApplicable, match="Train data is unavailable"):
        get_preprocessed_X(report, data_source="train")


def test_get_preprocessed_X_raises_not_applicable_for_sparse_features():
    """Sparse features cannot be normalized into a dense dataframe for analysis."""
    X_sparse = sp.random(20, 3, density=0.5, format="csr")
    y = np.random.default_rng(0).normal(size=20)
    report = MockReport(
        LinearRegression().fit(X_sparse.toarray(), y),
        X_train=X_sparse,
        y_train=y,
        X_test=X_sparse,
        y_test=y,
    )
    report._report_type = "estimator"
    with pytest.raises(CheckNotApplicable, match="Feature data is sparse"):
        get_preprocessed_X(report, data_source="train")


# collect_scores


def test_collect_scores_estimator_keyed_by_metric_identity(small_estimator_report):
    scores = collect_scores(small_estimator_report, data_source="test")
    assert ("R²", None, None, None) in scores
    timing_names = {"Fit time (s)", "Predict time (s)"}
    assert all(key[0] not in timing_names for key in scores)


def test_collect_scores_cv_report_averages_across_splits(small_cv_report):
    scores = collect_scores(small_cv_report, data_source="test")
    key = ("R²", None, None, None)
    assert key in scores
    per_split = small_cv_report.metrics.summarize(data_source="test").summary
    r2_rows = per_split[per_split["verbose_name"] == "R²"]
    assert scores[key]["score"] == pytest.approx(r2_rows["score"].mean())


def test_collect_scores_uses_classification_metrics(binary_classification_data):
    X, y = binary_classification_data
    report = EstimatorReport(
        LogisticRegression(),
        X_train=X[:80],
        y_train=y[:80],
        X_test=X[80:],
        y_test=y[80:],
    )
    scores = collect_scores(report, data_source="test")
    assert ("Accuracy", None, None, None) in scores


# _baseline_estimator_report


def test_baseline_estimator_report_unsupported_ml_task():
    """A report with an ml_task outside the supported list is not applicable."""
    stub = SimpleNamespace(ml_task="clustering", _report_type="estimator")
    with pytest.raises(CheckNotApplicable, match="Expected ML task to be one of"):
        _baseline_estimator_report(stub, kind="dummy")


def test_baseline_estimator_report_sparse_train_test_data():
    """Sparse train/test data cannot be normalized to build a baseline report."""
    X_sparse = sp.random(20, 3, density=0.5, format="csr")
    stub = SimpleNamespace(
        ml_task="regression",
        _report_type="estimator",
        X_train=X_sparse,
        X_test=X_sparse,
    )
    with pytest.raises(CheckNotApplicable, match="Data is sparse"):
        _baseline_estimator_report(stub, kind="dummy")


def test_baseline_estimator_report_train_data_missing(regression_train_test_split):
    """An estimator report without train data can't build any baseline."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    estimator = LinearRegression().fit(X_train, y_train)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    with pytest.raises(CheckNotApplicable, match="Train data is unavailable."):
        _baseline_estimator_report(report, kind="dummy")


def test_baseline_estimator_report_creation_fails_estimator(
    small_estimator_report, monkeypatch
):
    """Any exception while fitting the baseline report is caught."""

    def failing_fit(self, **kwargs):
        raise RuntimeError("Test error")

    monkeypatch.setattr(EstimatorReport, "_fit_estimator", failing_fit)
    with pytest.raises(CheckNotApplicable, match="Failed to create baseline report."):
        _baseline_estimator_report(small_estimator_report, kind="dummy")


def test_baseline_estimator_report_creation_fails_cv(small_cv_report, monkeypatch):
    """Any exception while fitting the baseline CV report is caught."""

    def failing_fit(self, **kwargs):
        raise RuntimeError("Test error")

    monkeypatch.setattr(EstimatorReport, "_fit_estimator", failing_fit)
    with pytest.raises(CheckNotApplicable, match="Failed to create baseline report."):
        _baseline_estimator_report(small_cv_report, kind="dummy")
