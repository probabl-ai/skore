"""
Common test for the metrics accessor of a ComparisonReport.
"""

import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

from skore import ComparisonReport, CrossValidationReport, EstimatorReport
from skore._displays import MetricsSummaryDisplay


@pytest.mark.parametrize("report", [EstimatorReport, CrossValidationReport])
def test_favorability_undefined_metrics(report):
    """Check that we don't introduce NaN when favorability is computed when
    for some estimators, the metric is undefined.

    Non-regression test for:
    https://github.com/probabl-ai/skore/issues/1755
    """

    X, y = make_classification(random_state=0)
    estimators = {"LinearSVC": LinearSVC(), "LogisticRegression": LogisticRegression()}

    if report is EstimatorReport:
        reports = {
            name: EstimatorReport(
                est, X_train=X, X_test=X, y_train=y, y_test=y, pos_label=1
            )
            for name, est in estimators.items()
        }
    else:
        reports = {
            name: CrossValidationReport(est, X=X, y=y, pos_label=1)
            for name, est in estimators.items()
        }

    comparison_report = ComparisonReport(reports)
    metrics = comparison_report.metrics.summarize()
    assert isinstance(metrics, MetricsSummaryDisplay)
    metrics_df = metrics.frame(flat_index=False, favorability=True, verbose_name=True)

    assert "Brier score" in metrics_df.index.get_level_values("Metric").to_numpy()
    assert "Favorability" in metrics_df.columns
    assert not metrics_df["Favorability"].isna().any()
    expected_values = {"(↗︎)", "(↘︎)"}
    actual_values = set(metrics_df["Favorability"].to_numpy())
    assert actual_values.issubset(expected_values)


@pytest.mark.parametrize("report", [EstimatorReport, CrossValidationReport])
def test_available_union(report):
    X, y = make_classification(random_state=0)
    estimators = {"LinearSVC": LinearSVC(), "LogisticRegression": LogisticRegression()}

    if report is EstimatorReport:
        reports = {
            name: EstimatorReport(
                est, X_train=X, X_test=X, y_train=y, y_test=y, pos_label=1
            )
            for name, est in estimators.items()
        }
    else:
        reports = {
            name: CrossValidationReport(est, X=X, y=y, pos_label=1)
            for name, est in estimators.items()
        }

    comparison_report = ComparisonReport(reports)
    available = comparison_report.metrics.available()
    expected = list(
        dict.fromkeys(
            metric
            for sub_report in comparison_report.reports_.values()
            for metric in sub_report.metrics.available()
        )
    )

    assert available == expected


@pytest.mark.parametrize("report", [EstimatorReport, CrossValidationReport])
def test_available_for_single_report_name(report):
    X, y = make_classification(random_state=0)
    estimators = {"LinearSVC": LinearSVC(), "LogisticRegression": LogisticRegression()}

    if report is EstimatorReport:
        reports = {
            name: EstimatorReport(
                est, X_train=X, X_test=X, y_train=y, y_test=y, pos_label=1
            )
            for name, est in estimators.items()
        }
    else:
        reports = {
            name: CrossValidationReport(est, X=X, y=y, pos_label=1)
            for name, est in estimators.items()
        }

    comparison_report = ComparisonReport(reports)
    assert (
        comparison_report.metrics.available(report_name="LinearSVC")
        == reports["LinearSVC"].metrics.available()
    )


@pytest.mark.parametrize("report", [EstimatorReport, CrossValidationReport])
def test_available_with_unknown_report_name_raises(report):
    X, y = make_classification(random_state=0)
    estimators = {"LinearSVC": LinearSVC(), "LogisticRegression": LogisticRegression()}

    if report is EstimatorReport:
        reports = {
            name: EstimatorReport(
                est, X_train=X, X_test=X, y_train=y, y_test=y, pos_label=1
            )
            for name, est in estimators.items()
        }
    else:
        reports = {
            name: CrossValidationReport(est, X=X, y=y, pos_label=1)
            for name, est in estimators.items()
        }

    comparison_report = ComparisonReport(reports)
    with pytest.raises(ValueError, match="Unknown report name"):
        comparison_report.metrics.available(report_name="unknown")


def test_non_default_n_jobs():
    """summarize() must work when ComparisonReport uses process-based parallelism.

    Joblib must not pickle a bound metrics-accessor method; that used to recurse
    while unpickling ``__getattr__`` / ``available()`` in worker processes.
    """
    X, y = make_classification(random_state=0)
    reports = {
        "LinearSVC": EstimatorReport(
            LinearSVC(), X_train=X, X_test=X, y_train=y, y_test=y, pos_label=1
        ),
        "LogisticRegression": EstimatorReport(
            LogisticRegression(),
            X_train=X,
            X_test=X,
            y_train=y,
            y_test=y,
            pos_label=1,
        ),
    }
    comparison_report = ComparisonReport(reports, n_jobs=2)
    display = comparison_report.metrics.summarize()

    assert isinstance(display, MetricsSummaryDisplay)
    assert set(display.summary["estimator"]) == {"LinearSVC", "LogisticRegression"}


@pytest.mark.parametrize("report_cls", [EstimatorReport, CrossValidationReport])
def test_metric_names_keep_their_case(report_cls):
    """Metric names are stored and displayed exactly as given.

    Non-regression test for:
    https://github.com/probabl-ai/skore/issues/3275
    """
    X, y = make_classification(n_samples=30, random_state=0)

    def MY_METRIC(estimator, X, y):
        return accuracy_score(estimator.predict(X), y)

    def My_metric(estimator, X, y):
        return accuracy_score(estimator.predict(X), y)

    def make_report(metric, *, verbose_name=None):
        estimator = LogisticRegression(max_iter=1_000)
        if report_cls is EstimatorReport:
            report = report_cls(estimator, X_train=X, y_train=y, X_test=X, y_test=y)
        else:
            report = report_cls(estimator, X, y, splitter=2)
        report.metrics.add(metric, verbose_name=verbose_name)
        return report

    report = make_report(MY_METRIC)
    frame = report.metrics.summarize().frame()
    assert "MY_METRIC" in frame.index
    assert "my_metric" not in frame.index
    verbose_names = set(
        report.metrics.summarize(metric="MY_METRIC").summary["verbose_name"]
    )
    assert verbose_names == {"MY_METRIC"}

    named = make_report(MY_METRIC, verbose_name="My metric")
    named_verbose_names = set(
        named.metrics.summarize(metric="MY_METRIC").summary["verbose_name"]
    )
    assert named_verbose_names == {"My metric"}

    comparison = ComparisonReport([report, make_report(My_metric)])
    compared = comparison.metrics.summarize().frame()
    assert "MY_METRIC" in compared.index
    assert "My_metric" in compared.index
    assert "my_metric" not in compared.index
