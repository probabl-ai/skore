import jedi
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error

from skore import CrossValidationReport, EstimatorReport


@pytest.fixture(
    params=[
        "estimator_reports_regression",
        "cross_validation_reports_regression",
        "comparison_estimator_reports_regression",
        "comparison_cross_validation_reports_regression",
    ]
)
def report(request):
    report = request.getfixturevalue(request.param)
    if isinstance(report, tuple):
        report = report[0]
    return report


def test_ipython_completion(report):
    """Non-regression test for #2386.

    We get no tab completions from IPython if jedi raises an exception, so we
    check here that jedi can produce completions without errors.
    """
    interp = jedi.Interpreter("r.", [{"r": report}])
    interp.complete(line=1, column=2)


def test_summarize_single_list_equivalence(report):
    """Passing a single string is equivalent to passing a list with one element."""
    display_single = report.metrics.summarize(metric="r2")
    display_list = report.metrics.summarize(metric=["r2"])
    assert_frame_equal(display_single.summary, display_list.summary)


def test_metrics_available_returns_metric_keys(report):
    metrics = report.metrics.available()

    assert isinstance(metrics, list)
    assert metrics
    assert all(isinstance(metric, str) for metric in metrics)


def test_metrics_available_updates_after_add(report):
    metrics_before = set(report.metrics.available())
    scorer = make_scorer(
        mean_squared_error, greater_is_better=False, response_method="predict"
    )
    report.metrics.add(scorer)

    metrics_after = set(report.metrics.available())
    assert metrics_before.issubset(metrics_after)
    assert "mean_squared_error" in metrics_after


def test_metrics_add_scorer(report):
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    report.metrics.add(scorer)

    display = report.metrics.summarize()
    assert "mean_squared_error" in display.summary["verbose_name"].values


def test_metrics_failure(report):
    """If a metric fails, `summarize` still returns."""

    def fail(estimator, X, y):
        raise Exception("test error")

    report.metrics.add(fail)

    display = report.metrics.summarize()

    assert "fail" in set(display.summary["verbose_name"])
    assert (
        display.summary[display.summary["verbose_name"] == "fail"]["score"].isna().all()
    )

    err_msg = r"Metric 'fail' has failed: Exception\('test error'\)"
    with pytest.warns(UserWarning, match=err_msg):
        display.frame()
        display.frame(flat_index=False)
    assert display.summary["name"].str.contains("fail", case=False).any()


@pytest.mark.parametrize("container", ["pandas", "polars"])
@pytest.mark.parametrize("report_cls", [EstimatorReport, CrossValidationReport])
def test_colliding_target_column_uses_stable_suffix(report_cls, container):
    """A feature named like the target is renamed with ``__y``, not ``_right``."""
    X, y = make_regression(n_samples=40, n_features=4, random_state=0)
    names = ["target", "f1", "f2", "f3"]
    if container == "pandas":
        features = pd.DataFrame(X, columns=names)
        target = pd.Series(y, name="target")
    else:
        features = pl.DataFrame({name: X[:, i] for i, name in enumerate(names)})
        target = pl.Series("target", y)

    if report_cls is EstimatorReport:
        report = EstimatorReport(
            LinearRegression(),
            X_train=features,
            y_train=target,
            X_test=features,
            y_test=target,
        )
        summarize_kwargs = ({}, {"data_source": "test"})
    else:
        report = CrossValidationReport(LinearRegression(), features, target, splitter=2)
        summarize_kwargs = ({},)

    expected = ["target", "f1", "f2", "f3", "target__y"]
    for kwargs in summarize_kwargs:
        columns = list(
            report.data.summarize(with_plots=False, **kwargs)
            .summary["dataframe"]
            .columns
        )
        assert columns == expected
        assert "target_right" not in columns
