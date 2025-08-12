from __future__ import annotations

from pydantic import ValidationError
from pytest import mark, param, raises
from skore_hub_project.metric import (
    PrecisionTest,
    PrecisionTestMean,
    PrecisionTestStd,
    PrecisionTrain,
    PrecisionTrainMean,
    PrecisionTrainStd,
)


@mark.parametrize(
    "report,Metric,name,verbose_name,greater_is_better,data_source,position,value",
    (
        param(
            "binary_classification",
            PrecisionTrain,
            "precision",
            "Precision (macro)",
            True,
            "train",
            None,
            1.0,
            id="PrecisionTrain",
        ),
        param(
            "binary_classification",
            PrecisionTest,
            "precision",
            "Precision (macro)",
            True,
            "test",
            None,
            0.8888888888888888,
            id="PrecisionTest",
        ),
        param(
            "cv_binary_classification",
            PrecisionTrainMean,
            "precision_mean",
            "Precision (macro) - MEAN",
            True,
            "train",
            None,
            1.0,
            id="PrecisionTrainMean",
        ),
        param(
            "cv_binary_classification",
            PrecisionTestMean,
            "precision_mean",
            "Precision (macro) - MEAN",
            True,
            "test",
            None,
            0.9343434343434345,
            id="PrecisionTestMean",
        ),
        param(
            "cv_binary_classification",
            PrecisionTrainStd,
            "precision_std",
            "Precision (macro) - STD",
            False,
            "train",
            None,
            0.0,
            id="PrecisionTrainStd",
        ),
        param(
            "cv_binary_classification",
            PrecisionTestStd,
            "precision_std",
            "Precision (macro) - STD",
            False,
            "test",
            None,
            0.045173090454541195,
            id="PrecisionTestStd",
        ),
    ),
)
def test_precision(
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
    monkeypatch.delattr(report.metrics.__class__, "precision")

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
