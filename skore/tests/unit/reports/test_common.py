import inspect
from functools import partial

import jedi
import pytest
import sklearn.metrics
from pandas.testing import assert_frame_equal
from sklearn.metrics import make_scorer, mean_squared_error

from skore._utils.docscrape import docstring_summary


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
    assert "Mean Squared Error" in display.summary["verbose_name"].values


def test_metrics_failure(report):
    """If a metric fails, `summarize` still returns."""

    def fail(estimator, X, y):
        raise Exception("test error")

    report.metrics.add(fail)

    display = report.metrics.summarize()

    assert "Fail" in set(display.summary["verbose_name"])
    assert (
        display.summary[display.summary["verbose_name"] == "Fail"]["score"].isna().all()
    )

    err_msg = r"Metric 'fail' has failed: Exception\('test error'\)"
    with pytest.warns(UserWarning, match=err_msg):
        display.frame()
        display.frame(flat_index=False)
    assert display.summary["name"].str.contains("fail", case=False).any()


def test_help_custom_metric(report, capsys):
    """Custom metrics are shown in the help menu, unless their name is not a valid
    identifier. Descriptions come from the metric docstring summary."""

    def custom(e, X, y):
        """Custom score used in help."""
        return 1

    report.metrics.add(custom)

    # Not a valid identifier
    report.metrics.add(lambda e, X, y: 2, name="a b")

    report.metrics.help()

    stdout = capsys.readouterr().out

    # Sanity check that help menu is there
    assert "predict_time(...)" in stdout
    assert "custom(...)" in stdout
    assert "Custom score used in help." in stdout
    # Built-in registry metrics also appear with a real description
    assert "accuracy(...)" in stdout or "r2(...)" in stdout
    # Not a valid identifier, so the help showing ".a b()" would be misleading
    assert "a b" not in stdout
    assert "Custom metric." not in stdout


def test_help_builtin_metric_description(report):
    """Built-in dynamic metrics expose a docstring-derived help description."""
    help_data = report.metrics._build_help_data()
    by_name = {method.name: method.description for method in help_data.methods}

    if "accuracy" in report.metrics.available():
        assert "accuracy" in by_name
        assert by_name["accuracy"] != "Custom metric."
        assert by_name["accuracy"] != "Registered metric."
    if "r2" in report.metrics.available():
        assert "r2" in by_name
        assert by_name["r2"] != "Custom metric."
        assert by_name["r2"] != "Registered metric."


def test_dynamic_metric_docstring_and_signature(report):
    """Dynamic metrics expose a constructed docstring and the accessor signature."""
    available = report.metrics.available()
    if "r2" in available:
        name = "r2"
        score_function = sklearn.metrics.r2_score
    elif "accuracy" in available:
        name = "accuracy"
        score_function = sklearn.metrics.accuracy_score
    else:
        pytest.skip("No built-in score-function metric available on this report")

    method = getattr(report.metrics, name)
    assert method.__name__ == name
    assert method.__doc__ is not None

    expected_summary = docstring_summary(score_function.__doc__)
    assert docstring_summary(method.__doc__) == expected_summary
    assert "data_source" in method.__doc__
    assert (
        "y_true" not in method.__doc__.split("Parameters", 1)[1].split("Returns", 1)[0]
    )
    assert (
        "y_pred" not in method.__doc__.split("Parameters", 1)[1].split("Returns", 1)[0]
    )

    assert inspect.signature(method) == inspect.signature(
        partial(report.metrics.get, name)
    )

    report_type = getattr(report, "_report_type", "")
    if "cross-validation" in report_type or "comparison" in report_type:
        assert "aggregate" in method.__doc__
    else:
        assert (
            "aggregate"
            not in method.__doc__.split("Parameters", 1)[1].split("Returns", 1)[0]
        )


def test_dynamic_metric_docstring_includes_metric_kwargs(report):
    """Metric kwargs defaults (e.g. average) appear in the constructed docstring."""
    available = report.metrics.available()
    if "precision" in available:
        name = "precision"
        expected_kwarg = "average"
    elif "r2" in available:
        name = "r2"
        expected_kwarg = "multioutput"
    else:
        pytest.skip("No metric with kwargs defaults available on this report")

    doc = getattr(report.metrics, name).__doc__
    assert doc is not None
    assert expected_kwarg in doc


def test_dynamic_custom_metric_docstring(report):
    """Custom metrics expose their callable summary and registry name."""

    def custom(e, X, y):
        """Custom score used as attribute docstring."""
        return 1

    report.metrics.add(custom)
    method = report.metrics.custom

    assert method.__name__ == "custom"
    assert (
        docstring_summary(method.__doc__) == "Custom score used as attribute docstring."
    )
