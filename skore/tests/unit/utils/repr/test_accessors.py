"""Integration tests for report accessor reprs."""

import pytest


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


@pytest.fixture(
    params=[
        "estimator_reports_regression",
        "cross_validation_reports_regression",
    ]
)
def report_with_data(request):
    report = request.getfixturevalue(request.param)
    if isinstance(report, tuple):
        report = report[0]
    return report


def test_inspection_accessor_repr(report):
    """Inspection accessor repr shows help panel content."""
    accessor = report.inspection
    help_data = accessor._build_help_data()
    repr_str = repr(accessor)
    html = accessor._repr_html_()

    assert help_data.title in repr_str
    assert help_data.root_node in repr_str
    assert "skore-accessor-help-" in html
    assert help_data.title in html
    for method in help_data.methods:
        assert method.name in repr_str
        assert method.name in html


def test_data_accessor_repr(report_with_data):
    """Data accessor repr shows help panel content."""
    accessor = report_with_data.data
    help_data = accessor._build_help_data()
    repr_str = repr(accessor)
    html = accessor._repr_html_()

    assert help_data.title in repr_str
    assert help_data.root_node in repr_str
    assert "skore-accessor-help-" in html
    assert help_data.title in html
    for method in help_data.methods:
        assert method.name in repr_str
        assert method.name in html


def _metrics_summary_frame(metrics_accessor):
    if hasattr(metrics_accessor, "_formatted_summary_frame"):
        return metrics_accessor._formatted_summary_frame()
    return metrics_accessor.summarize().frame()


def test_metrics_accessor_repr(report):
    """Metrics accessor __repr__ shows the summary frame and hints."""
    frame_repr = repr(_metrics_summary_frame(report.metrics))
    result = repr(report.metrics)

    assert result.startswith("Metrics summary:")
    assert frame_repr in result
    assert "Use .frame() to control the format of the output." not in result
    assert result.endswith("Explore available methods with .help().")


def test_metrics_accessor_repr_html(report):
    """Metrics accessor _repr_html_ shows the summary frame and hints."""
    frame_html = _metrics_summary_frame(report.metrics)._repr_html_()
    html = report.metrics._repr_html_()

    assert html.startswith("<p>Metrics summary:</p>")
    assert frame_html in html
    assert "Use .frame() to control the format of the output." not in html
    assert "Explore available methods with <code>.help()</code>." in html


def test_checks_accessor_repr(report):
    """Checks accessor repr shows checks summary frame content."""
    checks = report.checks
    frame = checks.summarize().frame()
    repr_str = repr(checks)
    html = checks._repr_html_()

    for row in frame.itertuples():
        assert row.code in repr_str
        assert row.title in repr_str
        assert row.code in html
        assert row.title in html
    assert "Fast mode is on" in html
    assert "Mute a check by passing" in html
