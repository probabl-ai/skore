import numpy as np
import pytest

from skore import CrossValidationReport


def test_cache_key_with_string_aggregate_is_not_split(
    forest_binary_classification_data,
):
    """
    Check that summarize results are properly cached.
    Non-regression test for: https://github.com/probabl-ai/skore/issues/2450
    Note: aggregate is now a parameter of frame(), not summarize(), so it's not
    part of the cache key. The caching happens at the summarize level.
    """
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)

    report.metrics.summarize().frame(aggregate="mean")

    summarize_cache_keys = [key for key in report._cache if key[1] == "summarize"]
    assert summarize_cache_keys, "Summarize results should be cached"


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_summarize_pos_label_overwrite(metric, logistic_binary_classification_data):
    """Check that `pos_label` can be overwritten in `summarize`"""
    classifier, X, y = logistic_binary_classification_data
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]

    report = CrossValidationReport(classifier, X, y)
    result_both_labels = report.metrics.summarize(metric=metric).frame().reset_index()
    assert result_both_labels["Label / Average"].to_list() == ["A", "B"]
    result_both_labels = result_both_labels.set_index(["Metric", "Label / Average"])

    report = CrossValidationReport(classifier, X, y, pos_label="B")
    result = report.metrics.summarize(metric=metric).frame().reset_index()
    assert "Label / Average" not in result.columns
    result = result.set_index("Metric")
    assert (
        result.loc[metric.capitalize(), (report.estimator_name_, "mean")]
        == result_both_labels.loc[
            (metric.capitalize(), "B"), (report.estimator_name_, "mean")
        ]
    )

    result = (
        report.metrics.summarize(metric=metric, pos_label="A").frame().reset_index()
    )
    assert "Label / Average" not in result.columns
    result = result.set_index("Metric")
    assert (
        result.loc[metric.capitalize(), (report.estimator_name_, "mean")]
        == result_both_labels.loc[
            (metric.capitalize(), "A"), (report.estimator_name_, "mean")
        ]
    )
