from __future__ import annotations

from numpy.testing import assert_almost_equal
from pydantic import ValidationError
from pytest import mark, param, raises
from skore_hub_project.metric import (
    RmseTest,
    RmseTestMean,
    RmseTestStd,
    RmseTrain,
    RmseTrainMean,
    RmseTrainStd,
)


@mark.parametrize(
    "report,Metric,name,verbose_name,greater_is_better,data_source,position,value",
    (
        param(
            "regression",
            RmseTrain,
            "rmse",
            "RMSE",
            False,
            "train",
            3,
            2.4355616448506994,
            id="RmseTrain",
        ),
        param(
            "regression",
            RmseTest,
            "rmse",
            "RMSE",
            False,
            "test",
            3,
            73.15561429220227,
            id="RmseTest",
        ),
        param(
            "cv_regression",
            RmseTrainMean,
            "rmse_mean",
            "RMSE - MEAN",
            False,
            "train",
            3,
            2.517350366068551,
            id="RmseTrainMean",
        ),
        param(
            "cv_regression",
            RmseTestMean,
            "rmse_mean",
            "RMSE - MEAN",
            False,
            "test",
            3,
            61.4227542951946,
            id="RmseTestMean",
        ),
        param(
            "cv_regression",
            RmseTrainStd,
            "rmse_std",
            "RMSE - STD",
            False,
            "train",
            None,
            0.19476154817956934,
            id="RmseTrainStd",
        ),
        param(
            "cv_regression",
            RmseTestStd,
            "rmse_std",
            "RMSE - STD",
            False,
            "test",
            None,
            12.103352184447452,
            id="RmseTestStd",
        ),
    ),
)
def test_rmse(
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
    monkeypatch.delattr(report.metrics.__class__, "rmse")

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
