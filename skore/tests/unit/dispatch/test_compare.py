import pytest
from sklearn.linear_model import LinearRegression

from skore import ComparisonReport, compare, evaluate


@pytest.mark.parametrize("n_jobs", [None, 2])
def test_compare_returns_comparison_report(regression_data, n_jobs):
    """Calling compare with at least two reports returns a ComparisonReport."""
    X, y = regression_data
    report_1 = evaluate(LinearRegression(), X, y, splitter=0.2)
    report_2 = evaluate(LinearRegression(), X, y, splitter=0.2)
    result = compare([report_1, report_2], n_jobs=n_jobs)
    assert isinstance(result, ComparisonReport)
