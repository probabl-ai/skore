"""Tests for MetricsSummaryDisplay repr."""

import pytest

from skore import CrossValidationReport, EstimatorReport


def test_repr_includes_frame_and_hint(forest_binary_classification_with_test):
    """Check that __repr__ shows the default frame and a trailing hint."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    display = EstimatorReport(
        estimator, X_test=X_test, y_test=y_test
    ).metrics.summarize()

    repr_str = repr(display)
    assert repr_str.startswith(repr(display.frame()))
    assert repr_str.endswith("Use .frame() to control the format of the output.")
    assert (
        "Use .plot() to plot the data" not in display._repr_mimebundle_()["text/plain"]
    )


def test_repr_html_includes_frame_and_hint(forest_binary_classification_with_test):
    """Check that _repr_html_ shows the default frame and a trailing hint."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    display = EstimatorReport(
        estimator, X_test=X_test, y_test=y_test
    ).metrics.summarize()

    html = display._repr_html_()
    assert html.startswith(display.frame()._repr_html_())
    assert "Use <code>.frame()</code> to control the format of the output." in html
    mime_html = display._repr_mimebundle_()["text/html"]
    assert "data:image/png;base64," not in mime_html
    assert "Use <code>.plot()</code> to control the view" not in mime_html


@pytest.fixture(params=["estimator", "cross-validation"])
def display_fail(request):
    estimator, X_test, y_test = request.getfixturevalue(
        "forest_binary_classification_with_test"
    )

    if request.param == "estimator":
        report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    elif request.param == "cross-validation":
        report = CrossValidationReport(estimator, X=X_test, y=y_test)
    else:
        raise ValueError(request.param)

    def fail1(estimator, X, y):
        raise Exception("Fail 1")

    def fail2(estimator, X, y):
        raise Exception("Fail 2")

    report.metrics.add(fail1)
    report.metrics.add(fail2)

    display = report.metrics.summarize()
    return display


@pytest.mark.filterwarnings(r"ignore:Metric 'fail\d' has failed:UserWarning")
def test_repr_failure(display_fail):
    """Check that __repr__ shows failed metrics."""
    repr_str = repr(display_fail)
    assert repr_str.startswith(repr(display_fail.frame()))

    # NaN is not filtered out of the dataframe
    assert "Fail" in repr(display_fail.frame())
    assert "Fail" in repr(display_fail.frame(aggregate=None))

    assert "Use .frame() to control the format of the output." in repr_str
    assert (
        "Use .plot() to plot the data"
        not in display_fail._repr_mimebundle_()["text/plain"]
    )

    # fail2 was added last, so it appears first in the DataFrame
    assert repr_str.endswith(
        "\nMetric 'fail2' has failed: Exception('Fail 2')"
        "\nMetric 'fail1' has failed: Exception('Fail 1')"
    )

    # Error messages are not duplicated (for CV)
    assert repr_str.count("'fail1'") == 1
    assert repr_str.count("'fail2'") == 1


@pytest.mark.filterwarnings(r"ignore:Metric 'fail\d' has failed:UserWarning")
def test_repr_html_failure(display_fail):
    """Check that _repr_html_ shows failed metrics."""
    repr_html = display_fail._repr_html_()
    assert repr_html.startswith(display_fail.frame()._repr_html_())

    # NaN is not filtered out of the dataframe
    assert "Fail" in display_fail.frame()._repr_html_()
    assert "Fail" in display_fail.frame(aggregate=None)._repr_html_()

    assert "Use <code>.frame()</code> to control the format of the output." in repr_html
    mime_html = display_fail._repr_mimebundle_()["text/html"]
    assert "data:image/png;base64," not in mime_html
    assert "Use <code>.plot()</code> to control the view" not in mime_html

    # fail2 was added last, so it appears first in the DataFrame
    assert repr_html.endswith(
        "<p role=\"note\">Metric 'fail2' has failed: Exception('Fail 2')</p>"
        "<p role=\"note\">Metric 'fail1' has failed: Exception('Fail 1')</p>"
    )

    # Error messages are not duplicated (for CV)
    assert repr_html.count("'fail1'") == 1
    assert repr_html.count("'fail2'") == 1
