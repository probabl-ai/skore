import jedi
import pytest
from pandas.testing import assert_frame_equal
from sklearn.metrics import make_scorer, mean_squared_error


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
