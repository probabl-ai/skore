import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from skore import EstimatorReport


@pytest.fixture
def linear_regression_with_test():
    X, y = make_regression(n_samples=100, n_features=5, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    regressor = LinearRegression().fit(X_train, y_train)
    return regressor, X_test, y_test


def test_metric_aliases_without_neg_prefix(linear_regression_with_test):
    """Check that metrics can be passed without the 'neg_' prefix and are
    automatically resolved to their 'neg_' prefixed sklearn scorer equivalent.
    This is the fix for https://github.com/probabl-ai/skore/issues/2607
    """
    regressor, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(regressor, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(
        metric=["mean_squared_error", "mean_absolute_error", "root_mean_squared_error"]
    )

    result = display.frame()
    assert isinstance(result, pd.DataFrame)
    assert "Mean Squared Error" in result.index
    assert "Mean Absolute Error" in result.index
    assert "Root Mean Squared Error" in result.index


def test_metric_aliases_same_result_as_neg_prefix(linear_regression_with_test):
    """Check that passing 'mean_squared_error' gives the same score as
    passing 'neg_mean_squared_error'.
    """
    regressor, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(regressor, X_test=X_test, y_test=y_test)

    display_with_neg = report.metrics.summarize(metric=["neg_mean_squared_error"])
    display_without_neg = report.metrics.summarize(metric=["mean_squared_error"])

    score_with_neg = display_with_neg.frame().values[0][0]
    score_without_neg = display_without_neg.frame().values[0][0]

    assert score_with_neg == pytest.approx(score_without_neg)
