from __future__ import annotations

from pydantic import ValidationError
from pytest import mark, param, raises
from skore_hub_project.metric import (
    AccuracyTest,
    AccuracyTestMean,
    AccuracyTestStd,
    AccuracyTrain,
    AccuracyTrainMean,
    AccuracyTrainStd,
)


@mark.parametrize(
    "report,Metric,name,verbose_name,greater_is_better,data_source,position,value",
    (
        param(
            "binary_classification",
            AccuracyTrain,
            "accuracy",
            "Accuracy",
            True,
            "train",
            None,
            1.0,
            id="AccuracyTrain",
        ),
        param(
            "binary_classification",
            AccuracyTest,
            "accuracy",
            "Accuracy",
            True,
            "test",
            None,
            0.9,
            id="AccuracyTest",
        ),
        param(
            "cv_binary_classification",
            AccuracyTrainMean,
            "accuracy_mean",
            "Accuracy - MEAN",
            True,
            "train",
            None,
            1.0,
            id="AccuracyTrainMean",
        ),
        param(
            "cv_binary_classification",
            AccuracyTestMean,
            "accuracy_mean",
            "Accuracy - MEAN",
            True,
            "test",
            None,
            0.93,
            id="AccuracyTestMean",
        ),
        param(
            "cv_binary_classification",
            AccuracyTrainStd,
            "accuracy_std",
            "Accuracy - STD",
            False,
            "train",
            None,
            0.0,
            id="AccuracyTrainStd",
        ),
        param(
            "cv_binary_classification",
            AccuracyTestStd,
            "accuracy_std",
            "Accuracy - STD",
            False,
            "test",
            None,
            0.04472135954999579,
            id="AccuracyTestStd",
        ),
    ),
)
def test_accuracy(
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
    monkeypatch.delattr(report.metrics.__class__, "accuracy")

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
