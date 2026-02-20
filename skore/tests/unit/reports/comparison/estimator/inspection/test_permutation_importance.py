from sklearn.linear_model import LinearRegression

from skore import ComparisonReport, EstimatorReport, PermutationImportanceDisplay
from skore._utils._testing import check_cache_changed


def test_returns_display(regression_train_test_split):
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert hasattr(report.inspection, "permutation_importance")
    display = report.inspection.permutation_importance(seed=42, n_repeats=2)
    assert isinstance(display, PermutationImportanceDisplay)


def test_cache_behavior(regression_train_test_split):
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert report._cache == {}

    with check_cache_changed(report._cache):
        report.inspection.permutation_importance(seed=42, n_repeats=2)

    assert len(report._cache) == 1
