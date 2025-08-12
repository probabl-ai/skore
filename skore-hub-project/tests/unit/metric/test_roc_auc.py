from __future__ import annotations

from pydantic import ValidationError
from pytest import mark, param, raises
from skore_hub_project.metric import (
    RocAucTest,
    RocAucTestMean,
    RocAucTestStd,
    RocAucTrain,
    RocAucTrainMean,
    RocAucTrainStd,
)


@mark.parametrize(
    "report,Metric,name,verbose_name,greater_is_better,data_source,position,value",
    (
        param(
            "binary_classification",
            RocAucTrain,
            "roc_auc",
            "ROC AUC",
            True,
            "train",
            3,
            1.0,
            id="RocAucTrain",
        ),
        param(
            "binary_classification",
            RocAucTest,
            "roc_auc",
            "ROC AUC",
            True,
            "test",
            3,
            0.989010989010989,
            id="RocAucTest",
        ),
        param(
            "cv_binary_classification",
            RocAucTrainMean,
            "roc_auc_mean",
            "ROC AUC - MEAN",
            True,
            "train",
            None,
            1.0,
            id="RocAucTrainMean",
        ),
        param(
            "cv_binary_classification",
            RocAucTestMean,
            "roc_auc_mean",
            "ROC AUC - MEAN",
            True,
            "test",
            None,
            0.986,
            id="RocAucTestMean",
        ),
        param(
            "cv_binary_classification",
            RocAucTrainStd,
            "roc_auc_std",
            "ROC AUC - STD",
            False,
            "train",
            None,
            5.551115123125783e-17,
            id="RocAucTrainStd",
        ),
        param(
            "cv_binary_classification",
            RocAucTestStd,
            "roc_auc_std",
            "ROC AUC - STD",
            False,
            "test",
            None,
            0.015165750888103078,
            id="RocAucTestStd",
        ),
    ),
)
def test_roc_auc(
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
    monkeypatch.delattr(report.metrics.__class__, "roc_auc")

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
