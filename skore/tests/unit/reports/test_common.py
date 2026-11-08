import jedi
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error

from skore import ComparisonReport, CrossValidationReport, EstimatorReport


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


def _make_report(report_type, X, y):
    if report_type == "estimator":
        return EstimatorReport(
            LinearRegression(), X_train=X, y_train=y, X_test=X, y_test=y
        )
    if report_type == "cross_validation":
        return CrossValidationReport(LinearRegression(), X, y, splitter=2)
    if report_type == "comparison_estimator":
        return ComparisonReport(
            [
                EstimatorReport(
                    LinearRegression(), X_train=X, y_train=y, X_test=X, y_test=y
                )
                for _ in range(2)
            ]
        )
    return ComparisonReport(
        [CrossValidationReport(LinearRegression(), X, y, splitter=2) for _ in range(2)]
    )


def _coefficient_features(report):
    names = report.inspection.coefficients().frame()["feature"]
    return [name for name in pd.unique(names) if name != "Intercept"]


def _dataset_feature_columns(report):
    if isinstance(report, CrossValidationReport):
        display = report.data.summarize(with_plots=False)
    elif isinstance(report, EstimatorReport):
        display = report.data.summarize(data_source="test", with_plots=False)
    else:
        return None
    return list(display.summary["dataframe"].columns)[:3]


@pytest.mark.parametrize(
    "report_type",
    [
        "estimator",
        "cross_validation",
        "comparison_estimator",
        "comparison_cross_validation",
    ],
)
def test_numpy_fallback_feature_names(report_type):
    """NumPy inputs use ``Feature #{i}`` for every report type."""
    X, y = make_regression(n_samples=40, n_features=3, random_state=0)
    report = _make_report(report_type, X, y)
    expected = ["Feature #0", "Feature #1", "Feature #2"]
    assert _coefficient_features(report) == expected
    dataset_columns = _dataset_feature_columns(report)
    if dataset_columns is not None:
        assert dataset_columns == expected


@pytest.mark.parametrize(
    "report_type",
    [
        "estimator",
        "cross_validation",
        "comparison_estimator",
        "comparison_cross_validation",
    ],
)
def test_integer_column_names_are_stringified_on_a_copy(report_type):
    """Non-string column names are copied to strings before fit."""
    X, y = make_regression(n_samples=40, n_features=3, random_state=0)
    frame = pd.DataFrame(X)
    original_columns = list(frame.columns)
    report = _make_report(report_type, frame, y)

    assert list(frame.columns) == original_columns
    assert _coefficient_features(report) == ["0", "1", "2"]
    dataset_columns = _dataset_feature_columns(report)
    if dataset_columns is not None:
        assert dataset_columns == ["0", "1", "2"]


@pytest.mark.parametrize("container", ["pandas", "polars"])
@pytest.mark.parametrize(
    "report_type",
    [
        "estimator",
        "cross_validation",
        "comparison_estimator",
        "comparison_cross_validation",
    ],
)
def test_string_column_names_are_preserved(report_type, container):
    """String column names are stored and displayed unchanged."""
    X, y = make_regression(n_samples=40, n_features=3, random_state=0)
    names = ["f0", "f1", "f2"]
    if container == "pandas":
        features = pd.DataFrame(X, columns=names)
        target = y
    else:
        features = pl.DataFrame({name: X[:, i] for i, name in enumerate(names)})
        target = pl.Series(y)

    report = _make_report(report_type, features, target)
    assert _coefficient_features(report) == names
    dataset_columns = _dataset_feature_columns(report)
    if dataset_columns is not None:
        assert dataset_columns == names
