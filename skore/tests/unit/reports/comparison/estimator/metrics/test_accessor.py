def test_random_state(comparison_estimator_reports_regression):
    """If random_state is None (the default) the call should not be cached."""
    report = comparison_estimator_reports_regression
    report.metrics.prediction_error()
    # skore stores the predictions of the child estimator reports, but not the
    # concatenated comparison display.
    assert all(child_report._cache == {} for child_report in report.reports_.values())
