from __future__ import annotations

from numpy.testing import assert_almost_equal
from pydantic import ValidationError
from pytest import mark, param, raises
from skore_hub_project.metric import (
    R2Test,
    R2TestMean,
    R2TestStd,
    R2Train,
    R2TrainMean,
    R2TrainStd,
)


@mark.parametrize(
    "report,Metric,name,verbose_name,greater_is_better,data_source,position,value",
    (
        param(
            "regression",
            R2Train,
            "r2",
            "R²",
            True,
            "train",
            None,
            0.9997075936149707,
            id="R2Train",
        ),
        param(
            "regression",
            R2Test,
            "r2",
            "R²",
            True,
            "test",
            None,
            0.6757085221095596,
            id="R2Test",
        ),
        param(
            "cv_regression",
            R2TrainMean,
            "r2_mean",
            "R² - MEAN",
            True,
            "train",
            None,
            0.9996757990105992,
            id="R2TrainMean",
        ),
        param(
            "cv_regression",
            R2TestMean,
            "r2_mean",
            "R² - MEAN",
            True,
            "test",
            None,
            0.7999871077321583,
            id="R2TestMean",
        ),
        param(
            "cv_regression",
            R2TrainStd,
            "r2_std",
            "R² - STD",
            False,
            "train",
            None,
            4.8687384179271224e-05,
            id="R2TrainStd",
        ),
        param(
            "cv_regression",
            R2TestStd,
            "r2_std",
            "R² - STD",
            False,
            "test",
            None,
            0.0528129534702117,
            id="R2TestStd",
        ),
    ),
)
def test_r2(
    monkeypatch,
    report,
    Metric,
    name,
    verbose_name,
    greater_is_better,
    data_source,
    position,
    value,
    request,
):
    report = request.getfixturevalue(report)

    # available accessor
    metric = Metric(report=report).model_dump()
    metric_value = metric.pop("value")

    assert_almost_equal(metric_value, value)
    assert metric == {
        "name": name,
        "verbose_name": verbose_name,
        "greater_is_better": greater_is_better,
        "data_source": data_source,
        "position": position,
    }

    # unavailable accessor
    monkeypatch.delattr(report.metrics.__class__, "r2")

    assert Metric(report=report).model_dump() == {
        "name": name,
        "verbose_name": verbose_name,
        "greater_is_better": greater_is_better,
        "data_source": data_source,
        "position": position,
        "value": None,
    }

    # wrong type
    with raises(ValidationError):
        Metric(report=None)
