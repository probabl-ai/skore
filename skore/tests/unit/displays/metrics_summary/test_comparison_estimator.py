import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from skore import ComparisonReport, MetricsSummaryDisplay, evaluate


def test_format_auto_uses_long(estimator_reports_binary_classification):
    """Auto format uses long layout for comparison-estimator reports."""
    estimator_report_1, estimator_report_2 = estimator_reports_binary_classification
    report = ComparisonReport([estimator_report_1, estimator_report_2])

    result = report.metrics.summarize().frame(format="auto")

    assert isinstance(result.index, pd.RangeIndex)
    assert "estimator" in result.columns


def test_data_source_both(estimator_reports_binary_classification):
    """Check that `MetricsSummaryDisplay` works with `data_source="both"`."""
    estimator_report_1, estimator_report_2 = estimator_reports_binary_classification
    report = ComparisonReport([estimator_report_1, estimator_report_2])
    result = report.metrics.summarize(data_source="both").frame(format="wide")

    assert result.index.to_list() == [
        "score",
        "accuracy",
        "precision_0",
        "precision_1",
        "recall_0",
        "recall_1",
        "roc_auc",
        "log_loss",
        "brier_score",
        "fit_time",
        "predict_time",
    ]
    assert result.columns.to_list() == [
        "DummyClassifier_1 (train)",
        "DummyClassifier_1 (test)",
        "DummyClassifier_2 (train)",
        "DummyClassifier_2 (test)",
    ]


def test_format_wide(estimator_reports_binary_classification):
    """Compact format always returns a flat index and columns."""
    report_1, report_2 = estimator_reports_binary_classification
    report = ComparisonReport({"report_1": report_1, "report_2": report_2})
    result = report.metrics.summarize()
    assert isinstance(result, MetricsSummaryDisplay)
    result_df = result.frame(format="wide")
    assert isinstance(result_df.index, pd.Index)
    assert result_df.index.tolist() == [
        "score",
        "accuracy",
        "precision_0",
        "precision_1",
        "recall_0",
        "recall_1",
        "roc_auc",
        "log_loss",
        "brier_score",
        "fit_time",
        "predict_time",
    ]
    assert result_df.columns.tolist() == ["report_1", "report_2"]


def test_favorability(comparison_estimator_reports_binary_classification):
    """Check that the behaviour of `favorability` is correct."""
    report = comparison_estimator_reports_binary_classification
    display = report.metrics.summarize()
    result = display.frame(format="wide", favorability=True)
    assert set(result["Favorability"]) == {"(↗︎)", "(↘︎)"}


def test_frame_has_estimator_column(
    comparison_estimator_reports_binary_classification,
):
    """The tidy frame exposes an ``estimator`` column with each estimator name."""
    report = comparison_estimator_reports_binary_classification
    frame = report.metrics.summarize().frame(format="long")

    assert isinstance(frame.index, pd.RangeIndex)
    assert "estimator" in frame.columns
    assert "split" not in frame.columns
    assert frame["estimator"].nunique() == 2


def test_aggregate(comparison_estimator_reports_binary_classification):
    """Passing `aggregate` should have no effect, as this argument is only relevant
    when comparing `CrossValidationReport`s."""
    report = comparison_estimator_reports_binary_classification
    np.testing.assert_allclose(
        report.metrics.summarize().frame(format="wide", aggregate="mean"),
        report.metrics.summarize().frame(format="wide"),
    )


class TestDisambiguateMetrics:
    """Disambiguation of metrics with the same name."""

    @pytest.fixture
    def estimator_reports_regression(self, estimator_reports_regression):
        """Deep-copy estimators to make the tests immune to state pollution."""
        return copy.deepcopy(estimator_reports_regression)

    def test_score(self, regression):
        """Estimators with different `.score` source code render as Score_1, Score_2."""
        X, y = regression

        class A(DummyRegressor):
            def score(self, X, y):
                return 1

        class B(DummyRegressor):
            def score(self, X, y):
                return 2

        report = evaluate([A(), B()], X, y, splitter=0.2)

        result = report.metrics.summarize().frame(format="wide", verbose_name=True)

        metric_names = result.index.tolist()
        assert metric_names[0] == "Score_1"
        assert metric_names[-1] == "Score_2"
        assert "Score" not in metric_names

        assert result.loc["Score_1", "A"] == 1
        assert result.loc["Score_2", "B"] == 2
        assert np.isnan(result.loc["Score_1", "B"])
        assert np.isnan(result.loc["Score_2", "A"])

    def test_custom_metric(self, estimator_reports_regression):
        """Estimators with a custom metric with the same name but different source code
        render as Metric_1, Metric_2."""
        report_1, report_2 = estimator_reports_regression

        def metric(estimator, X, y):
            return 1

        report_1.metrics.add(metric)

        def metric(estimator, X, y):
            return 2

        report_2.metrics.add(metric)

        report = ComparisonReport([report_1, report_2])
        result = report.metrics.summarize().frame(format="wide", verbose_name=True)

        metric_names = result.index.tolist()
        assert metric_names[0] == "Metric_1"
        assert metric_names[-1] == "Metric_2"
        assert "Metric" not in metric_names

        assert result.loc["Metric_1", "DummyRegressor_1"] == 1
        assert result.loc["Metric_2", "DummyRegressor_2"] == 2
        assert np.isnan(result.loc["Metric_1", "DummyRegressor_2"])
        assert np.isnan(result.loc["Metric_2", "DummyRegressor_1"])

    def test_custom_metric_technical_name(self, estimator_reports_regression):
        """The technical name is disambiguated too, not only the verbose name.

        Two custom metrics sharing a technical ``name`` but with different source
        code (hence different fingerprints) render as ``metric_1`` and ``metric_2``
        in the default frame (``verbose_name=False``)."""
        report_1, report_2 = estimator_reports_regression

        def metric(estimator, X, y):
            return 1

        report_1.metrics.add(metric)

        def metric(estimator, X, y):
            return 2

        report_2.metrics.add(metric)

        report = ComparisonReport([report_1, report_2])
        result = report.metrics.summarize().frame(format="wide")

        metric_names = result.index.tolist()
        assert "metric_1" in metric_names
        assert "metric_2" in metric_names
        assert "metric" not in metric_names

        assert result.loc["metric_1", "DummyRegressor_1"] == 1
        assert result.loc["metric_2", "DummyRegressor_2"] == 2
        assert np.isnan(result.loc["metric_1", "DummyRegressor_2"])
        assert np.isnan(result.loc["metric_2", "DummyRegressor_1"])

    def test_avoids_existing_suffix(self, estimator_reports_regression):
        """Disambiguation suffixes skip over verbose names that are already taken.

        If there is already a metric called ``"Metric_1"``, then disambiguating two
        other custom metrics both called ``"Metric"`` must yield ``Metric_2``
        and ``Metric_3`` to avoid colliding with the pre-existing ``Metric_1``.
        """
        report_1, report_2 = estimator_reports_regression

        def metric_a(estimator, X, y):
            return 1

        def metric_b(estimator, X, y):
            return 2

        def metric_already_suffixed(estimator, X, y):
            return 3

        # Both reports get a custom metric with verbose_name="Metric" but different
        # source code, so they would naturally be renamed to Metric_1 and Metric_2.
        report_1.metrics.add(metric_a, verbose_name="Metric")
        report_2.metrics.add(metric_b, verbose_name="Metric")

        # but one report also has a metric already named "Metric_1"
        report_1.metrics.add(metric_already_suffixed, verbose_name="Metric_1")

        report = ComparisonReport([report_1, report_2])
        result = report.metrics.summarize().frame(format="wide", verbose_name=True)

        # so the renaming must skip _1 and use _2 / _3.
        metric_names = result.index.tolist()
        assert "Metric" not in metric_names
        assert "Metric_1" in metric_names
        assert "Metric_2" in metric_names
        assert "Metric_3" in metric_names

        # The pre-existing "Metric_1" (metric_already_suffixed) is left untouched
        assert result.loc["Metric_1", "DummyRegressor_1"] == 3

        # The two ambiguous "Metric"s get _2 and _3 in row-appearance order.
        assert result.loc["Metric_2", "DummyRegressor_1"] == 1
        assert result.loc["Metric_3", "DummyRegressor_2"] == 2

    def test_redefined_builtin(self, estimator_reports_regression):
        """A custom metric reusing a built-in's verbose name is disambiguated.

        If one report and replaces a builtin metric with a custom callable with the same
        name, we can differentiate the builtin from the custom.
        """
        report_1, report_2 = estimator_reports_regression

        def custom_r2(estimator, X, y):
            return 1000

        # Replace the builtin R² with a custom metric with the same name
        report_1.metrics.remove("r2")
        report_1.metrics.add(custom_r2, verbose_name="R²")

        report = ComparisonReport([report_1, report_2])
        result = report.metrics.summarize().frame(format="wide", verbose_name=True)

        assert result.loc["R²_1", "DummyRegressor_1"] == 1000
        assert np.isnan(result.loc["R²_1", "DummyRegressor_2"])
        assert np.isnan(result.loc["R²_2", "DummyRegressor_1"])
        assert not np.isnan(result.loc["R²_2", "DummyRegressor_2"])

    def test_multimetric_scorer_submetric(self, estimator_reports_regression):
        """A multimetric scorer's submetric sharing a built-in's name is disambiguated.

        When a user adds a multimetric scorer containing a key matching a
        built-in metric (e.g. ``"R²"``), the submetric's rows
        inherit the parent scorer's fingerprint, while the built-in carries
        ``fingerprint=None``. The display renders them as ``R²_1`` (custom)
        and ``R²_2`` (built-in) so they don't silently collide.
        """
        report_1, report_2 = estimator_reports_regression

        def multimetric_scorer(estimator, X, y):
            # "R²" clashes with the built-in R² metric.
            return {"R²": 999, "Custom Submetric": 42}

        report_1.metrics.add(multimetric_scorer)

        report = ComparisonReport([report_1, report_2])
        result = report.metrics.summarize().frame(format="wide", verbose_name=True)

        # Custom metric, only computed for report 1
        assert result.loc["R²_1", "DummyRegressor_1"] == 999
        assert np.isnan(result.loc["R²_1", "DummyRegressor_2"])

        # Builtin metric computed for both
        assert not np.isnan(result.loc["R²_2", "DummyRegressor_1"])
        assert not np.isnan(result.loc["R²_2", "DummyRegressor_2"])

        # The other submetric only appears for report 1
        assert result.loc["Custom Submetric", "DummyRegressor_1"] == 42
        assert np.isnan(result.loc["Custom Submetric", "DummyRegressor_2"])
