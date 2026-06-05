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
    assert_frame_equal(display_single.data, display_list.data)


def test_metrics_repr(report):
    """Test that metrics accessor __repr__ shows the summary frame and hints."""
    frame_repr = repr(report.metrics.summarize().frame())
    result = repr(report.metrics)
    assert result.startswith("Metrics summary:")
    assert frame_repr in result
    assert "Use .frame() to control the format of the output." not in result
    assert result.endswith("Explore available methods with .help().")


def test_metrics_repr_html(report):
    """Test that metrics accessor _repr_html_ shows the summary frame and hints."""
    frame_html = report.metrics.summarize().frame()._repr_html_()
    html = report.metrics._repr_html_()
    assert html.startswith("<p>Metrics summary:</p>")
    assert frame_html in html
    assert "Use .frame() to control the format of the output." not in html
    assert "<code>.help()</code>" in html


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
    assert "Mean Squared Error" in display.data["metric_verbose_name"].values
