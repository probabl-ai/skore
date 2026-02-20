from sklearn.linear_model import Ridge

from skore import ComparisonReport, CrossValidationReport, PermutationImportanceDisplay
from skore._utils._testing import check_cache_changed


def test_returns_display(regression_data):
    X, y = regression_data
    report_1 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report_2 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert hasattr(report.inspection, "permutation_importance")
    display = report.inspection.permutation_importance(seed=42, n_repeats=2)
    assert isinstance(display, PermutationImportanceDisplay)


def test_split_column(regression_data):
    X, y = regression_data
    report_1 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report_2 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.inspection.permutation_importance(seed=42, n_repeats=2)
    assert set(display.importances["split"]) == {0, 1}


def test_cache_behavior(regression_data):
    X, y = regression_data
    report_1 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report_2 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert report._cache == {}

    with check_cache_changed(report._cache):
        report.inspection.permutation_importance(seed=42, n_repeats=2)

    assert len(report._cache) == 1
