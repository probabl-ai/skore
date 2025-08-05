from sklearn.linear_model import LinearRegression
from skore import CrossValidationReport


def test_help(capsys, regression_data):
    """Check that the help method writes to the console"""
    X, y = regression_data
    report = CrossValidationReport(LinearRegression(), X=X, y=y)
    report.feature_importance.help()
    captured = capsys.readouterr()

    assert "Available feature importance methods" in captured.out
    assert "coefficients" in captured.out

    report.feature_importance.coefficients().help()
    captured = capsys.readouterr()

    assert "frame" in captured.out
    assert "plot" in captured.out
    assert "set_style" in captured.out


def test_repr(regression_data):
    """Check that __repr__ returns a string starting with the expected prefix."""
    X, y = regression_data
    report = CrossValidationReport(LinearRegression(), X=X, y=y)

    repr_str = repr(report.feature_importance)
    assert "skore.CrossValidationReport.feature_importance" in repr_str
    assert "help()" in repr_str
