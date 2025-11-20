import pytest

from skore import EstimatorReport
from skore._sklearn._estimator.feature_importance_accessor import (
    _check_scoring,
    metric_to_scorer,
)


def test_feature_importance_help(capsys, linear_regression_with_test):
    """Check that the help method writes to the console."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    report.feature_importance.help()
    captured = capsys.readouterr()
    assert "Available feature importance methods" in captured.out
    assert "coefficients" in captured.out

    report.feature_importance.coefficients().help()
    captured = capsys.readouterr()
    assert "frame" in captured.out
    assert "plot" in captured.out
    assert "set_style" in captured.out


def test_feature_importance_repr(linear_regression_with_test):
    """Check that __repr__ returns a string starting with the expected prefix."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    repr_str = repr(report.feature_importance)
    assert "skore.EstimatorReport.feature_importance" in repr_str
    assert "help()" in repr_str


def _dummy_scorer(*_) -> float:
    return 0.0


@pytest.mark.parametrize(
    "scoring, expected",
    [
        pytest.param(None, None, id="none"),
        pytest.param(_dummy_scorer, _dummy_scorer, id="callable"),
        pytest.param({"custom": _dummy_scorer}, {"custom": _dummy_scorer}, id="dict"),
        pytest.param("rmse", {"rmse": metric_to_scorer["rmse"]}, id="single-metric"),
        pytest.param(
            ["r2", "rmse"],
            {"r2": metric_to_scorer["r2"], "rmse": metric_to_scorer["rmse"]},
            id="multiple-metrics",
        ),
    ],
)
def test_check_scoring_passthrough_and_conversion(scoring, expected):
    """Check that the scoring is passed through or converted to a dictionary of\
    scorers."""
    assert _check_scoring(scoring) == expected


@pytest.mark.parametrize(
    "scoring, err_msg",
    [
        ("neg_root_mean_squared_error", "If scoring is a string"),
        (["r2", metric_to_scorer["rmse"]], "must contain only strings"),
        (3, "scoring must be a string"),
    ],
)
def test_check_scoring_errors(scoring, err_msg):
    """Check that the scoring is converted to a dictionary of scorers."""
    with pytest.raises(TypeError, match=err_msg):
        _check_scoring(scoring)
