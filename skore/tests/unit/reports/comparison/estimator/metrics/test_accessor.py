def test_random_state(comparison_estimator_reports_regression):
    """If random_state is None (the default) the call should not be cached."""
    report = comparison_estimator_reports_regression
    report.metrics.prediction_error()

    assert all(
        len(child_report._cache) == 0 for child_report in report.reports_.values()
    )
