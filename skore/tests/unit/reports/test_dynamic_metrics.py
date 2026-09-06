import inspect
from functools import partial

import pytest
import sklearn.metrics

import skore._utils.repr.data as data_module
from skore._metrics import Metric
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


def _parameters_section(doc: str) -> str:
    return doc.split("Parameters", 1)[1].split("Returns", 1)[0]


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
    # All reports in this fixture are regression.
    assert "r2(...)" in stdout
    # Not a valid identifier, so the help showing ".a b()" would be misleading
    assert "a b" not in stdout
    # "Custom metric." was the previous hardcoded help fallback; ensure it does not
    # reappear when a docstring-derived description is available.
    assert "Custom metric." not in stdout
    # Registry callables are separated from registry management and displays.
    assert "Registry" in stdout
    assert "Metrics" in stdout
    assert "Displays" in stdout


def test_help_groups_separate_registry_metrics_displays(report):
    """Help data partitions methods into Registry / Metrics / Displays groups.

    Registry callables (and custom metrics) land in Metrics; management helpers
    in Registry; ordered display helpers in Displays.
    """

    def custom(e, X, y):
        """Custom score used in groups."""
        return 1

    report.metrics.add(custom)

    help_data = report.metrics._build_help_data()
    assert help_data.groups is not None
    by_name = {
        group.name: [m.name for m in group.methods] for group in help_data.groups
    }
    assert list(by_name) == ["Registry", "Metrics", "Displays"]

    assert by_name["Registry"] == ["available", "add", "remove", "get"]
    assert "custom" in by_name["Metrics"]
    assert "r2" in by_name["Metrics"]
    assert by_name["Displays"][0] == "summarize"
    assert "custom" not in by_name["Displays"]
    assert "available" not in by_name["Displays"]
    # Registry callables come before static score helpers in Metrics.
    assert by_name["Metrics"].index("custom") < by_name["Metrics"].index("timings")


def test_help_groups_cover_every_method(report):
    """Every method shown in the metrics help belongs to exactly one group."""

    def custom(e, X, y):
        """Custom score used in groups."""
        return 1

    report.metrics.add(custom)

    help_data = report.metrics._build_help_data()
    grouped_names = [m.name for group in help_data.groups for m in group.methods]

    assert sorted(grouped_names) == sorted(m.name for m in help_data.methods)
    assert len(grouped_names) == len(set(grouped_names))


def test_help_groups_computed_once(report, monkeypatch):
    """Registry metrics are grouped in a single pass, not grouped then regrouped."""
    calls = []
    original = data_module._build_method_groups

    def counting(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(data_module, "_build_method_groups", counting)
    report.metrics._build_help_data()

    assert len(calls) == 1


def test_help_groups_are_expanded_by_default(report):
    """HTML help unfolds Registry / Metrics / Displays without an extra click."""
    html = report.metrics._create_help_html()
    assert html.count('<input type="checkbox" class="toggle" checked>') >= 3


def test_help_builtin_metric_description(report):
    """Built-in dynamic metrics expose a docstring-derived help description."""
    help_data = report.metrics._build_help_data()
    by_name = {method.name: method.description for method in help_data.methods}

    assert "r2" in by_name
    assert by_name["r2"] != "Custom metric."
    assert by_name["r2"] != "Registered metric."


def test_help_dynamic_metric_has_no_docs_link(report):
    """Registry metrics have no Sphinx page; HTML help shows a tooltip only."""
    help_data = report.metrics._build_help_data()
    by_name = {method.name: method for method in help_data.methods}

    assert by_name["r2"].doc_url is None
    assert by_name["get"].doc_url is not None
    assert by_name["get"].doc_url.startswith("https://docs.skore.probabl.ai/")

    html = report.metrics._create_help_html()
    # Dynamic metric: tooltip span only (no docs link).
    assert '<span class="method-tooltip">r2<' in html
    assert by_name["r2"].description in html
    # Static method keeps its Sphinx link.
    assert f'href="{by_name["get"].doc_url}"' in html


def test_dynamic_metric_docstring_and_signature(report):
    """Dynamic metrics expose a constructed docstring and the accessor signature."""
    method = report.metrics.r2
    assert method.__name__ == "r2"
    assert method.__doc__ is not None

    expected_summary = docstring_summary(sklearn.metrics.r2_score.__doc__)
    assert docstring_summary(method.__doc__) == expected_summary
    assert "data_source" in method.__doc__
    assert "y_true" not in _parameters_section(method.__doc__)
    assert "y_pred" not in _parameters_section(method.__doc__)

    assert inspect.signature(method) == inspect.signature(
        partial(report.metrics.get, "r2")
    )

    report_type = getattr(report, "_report_type", "")
    if report_type == "estimator":
        assert "aggregate" not in _parameters_section(method.__doc__)
    else:
        assert "aggregate" in method.__doc__


def test_dynamic_metric_docstring_keeps_score_function_parameter(report):
    """Metric kwargs reuse the score function's type spec, but with skore's default."""
    doc = report.metrics.r2.__doc__

    assert "multioutput : {'raw_values', 'uniform_average', 'variance_weighted'}" in doc
    # skore registers r2 with multioutput="raw_values", not scikit-learn's default
    assert "default='raw_values'" in doc
    assert "default='uniform_average'" not in doc
    # the description keeps its structure instead of being flattened into one line
    assert "\n    'raw_values' :\n" in doc
    assert "\n    .. versionchanged:: 0.19\n" in doc


def test_dynamic_custom_metric_docstring(report):
    """Custom metrics expose their callable summary and registry name."""

    def custom(e, X, y):
        """My custom metric."""
        return 1

    report.metrics.add(custom)

    assert report.metrics.custom.__name__ == "custom"
    assert docstring_summary(report.metrics.custom.__doc__) == "My custom metric."


def test_dynamic_custom_metric_docstring_without_docstring(report):
    """An undocumented custom metric falls back to its verbose name."""
    report.metrics.add(lambda e, X, y: 1, name="nodoc", verbose_name="No doc metric")

    assert docstring_summary(report.metrics.nodoc.__doc__) == "No doc metric"
    assert report.metrics._metric_help_description("nodoc") == "No doc metric"
    # the docstring of the Metric class itself describes the machinery, not the metric
    assert "A metric that can compute a score from a report." not in (
        report.metrics.nodoc.__doc__
    )


def test_dynamic_custom_metric_docstring_from_partial(report):
    """A metric built from a partial does not expose the stdlib partial docstring."""

    def business_loss(estimator, X, y, cost):
        return 1

    report.metrics.add(partial(business_loss, cost=10), name="loss")

    assert docstring_summary(report.metrics.loss.__doc__) == "Loss"
    assert "partial application" not in report.metrics.loss.__doc__


def test_resolve_metric(report):
    """Each accessor resolves registry metrics from its own report structure."""
    for name in report.metrics.available():
        metric = report.metrics._resolve_metric(name)
        assert isinstance(metric, Metric)
        assert metric.name == name

    assert report.metrics._resolve_metric("does_not_exist") is None


def test_resolve_metric_comparison_partial_coverage(
    comparison_estimator_reports_regression,
):
    """A comparison resolves a metric registered on a single sub-report."""
    comparison = comparison_estimator_reports_regression
    first, *_ = comparison.reports_.values()
    first.metrics.add(lambda e, X, y: 1, name="only_first")

    assert comparison.metrics._resolve_metric("only_first").name == "only_first"
