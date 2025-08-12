from __future__ import annotations

from pydantic import ValidationError
from pytest import mark, param, raises
from skore_hub_project.metric import (
    LogLossTest,
    LogLossTestMean,
    LogLossTestStd,
    LogLossTrain,
    LogLossTrainMean,
    LogLossTrainStd,
)


@mark.parametrize(
    "report,Metric,name,verbose_name,greater_is_better,data_source,position,value",
    (
        param(
            "binary_classification",
            LogLossTrain,
            "log_loss",
            "Log loss",
            False,
            "train",
            4,
            0.06911280690412243,
            id="LogLossTrain",
        ),
        param(
            "binary_classification",
            LogLossTest,
            "log_loss",
            "Log loss",
            False,
            "test",
            4,
            0.3168690248138036,
            id="LogLossTest",
        ),
        param(
            "cv_binary_classification",
            LogLossTrainMean,
            "log_loss_mean",
            "Log loss - MEAN",
            False,
            "train",
            None,
            0.07706302057996325,
            id="LogLossTrainMean",
        ),
        param(
            "cv_binary_classification",
            LogLossTestMean,
            "log_loss_mean",
            "Log loss - MEAN",
            False,
            "test",
            None,
            0.23938382759923754,
            id="LogLossTestMean",
        ),
        param(
            "cv_binary_classification",
            LogLossTrainStd,
            "log_loss_std",
            "Log loss - STD",
            False,
            "train",
            None,
            0.003611541148136149,
            id="LogLossTrainStd",
        ),
        param(
            "cv_binary_classification",
            LogLossTestStd,
            "log_loss_std",
            "Log loss - STD",
            False,
            "test",
            None,
            0.030545861791452432,
            id="LogLossTestStd",
        ),
    ),
)
def test_log_loss(
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
    monkeypatch.delattr(report.metrics.__class__, "log_loss")

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
