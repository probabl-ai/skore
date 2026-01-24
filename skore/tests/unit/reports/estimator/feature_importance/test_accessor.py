import pytest

from skore._sklearn._estimator.feature_importance_accessor import (
    _check_metric,
    metric_to_scorer,
)


def _dummy_scorer(*_) -> float:
    return 0.0


@pytest.mark.parametrize(
    "metric, expected",
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
def test_check_metric_passthrough_and_conversion(metric, expected):
    """Check that the scoring is passed through or converted to a dictionary of\
    scorers."""
    assert _check_metric(metric) == expected


@pytest.mark.parametrize(
    "metric, err_msg",
    [
        ("neg_root_mean_squared_error", "If metric is a string"),
        (["r2", metric_to_scorer["rmse"]], "must contain only strings"),
        (3, "metric must be a string"),
    ],
)
def test_check_metric_errors(metric, err_msg):
    """Check that the metric is converted to a dictionary of scorers."""
    with pytest.raises(TypeError, match=err_msg):
        _check_metric(metric)
