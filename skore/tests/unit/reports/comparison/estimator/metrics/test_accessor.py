from skore._utils._testing import check_cache_unchanged


def test_seed_none(comparison_estimator_reports_regression):
    """If seed is None (the default) the call should not be cached."""
    report = comparison_estimator_reports_regression
    report.cache_predictions()
    with check_cache_unchanged([r._cache for r in report.reports_.values()]):
        report.metrics.prediction_error()
