from __future__ import annotations

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
            2.091309874874337e-13,
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
            73.92385036768957,
            id="RmseTest",
        ),
        param(
            "cv_regression",
            RmseTrainMean,
            "rmse_mean",
            "RMSE - MEAN",
            False,
            "train",
            None,
            2.471584795487725e-13,
            id="RmseTrainMean",
        ),
        param(
            "cv_regression",
            RmseTestMean,
            "rmse_mean",
            "RMSE - MEAN",
            False,
            "test",
            None,
            60.433160266994534,
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
            5.75989844775981e-14,
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
            9.96978706616252,
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
    assert Metric(report=report).model_dump() == {
        "name": name,
        "verbose_name": verbose_name,
        "greater_is_better": greater_is_better,
        "data_source": data_source,
        "position": position,
        "value": value,
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
