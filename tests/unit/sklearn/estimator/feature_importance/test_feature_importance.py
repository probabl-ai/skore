from sklearn.linear_model import LinearRegression
from skore import EstimatorReport


def test_estimator_report_feature_importance_help(capsys, regression_data):
    """Check that the help method writes to the console."""
    X, y = regression_data
    report = EstimatorReport(LinearRegression().fit(X, y), X_test=X, y_test=y)

    report.feature_importance.help()
    captured = capsys.readouterr()
    assert "Available feature importance methods" in captured.out
    assert "coefficients" in captured.out


def test_estimator_report_feature_importance_repr(regression_data):
    """Check that __repr__ returns a string starting with the expected prefix."""
    X, y = regression_data
    report = EstimatorReport(LinearRegression().fit(X, y), X_test=X, y_test=y)

    repr_str = repr(report.feature_importance)
    assert "skore.EstimatorReport.feature_importance" in repr_str
    assert "help()" in repr_str
