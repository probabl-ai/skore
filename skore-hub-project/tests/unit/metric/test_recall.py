from __future__ import annotations

from numpy.testing import assert_almost_equal
from pydantic import ValidationError
from pytest import mark, param, raises
from skore_hub_project.metric import (
    RecallTest,
    RecallTestMean,
    RecallTestStd,
    RecallTrain,
    RecallTrainMean,
    RecallTrainStd,
)


@mark.parametrize(
    "report,Metric,name,verbose_name,greater_is_better,data_source,position,value",
    (
        param(
            "binary_classification",
            RecallTrain,
            "recall",
            "Recall (macro)",
            True,
            "train",
            None,
            1.0,
            id="RecallTrain",
        ),
        param(
            "binary_classification",
            RecallTest,
            "recall",
            "Recall (macro)",
            True,
            "test",
            None,
            0.9230769230769231,
            id="RecallTest",
        ),
        param(
            "cv_binary_classification",
            RecallTrainMean,
            "recall_mean",
            "Recall (macro) - MEAN",
            True,
            "train",
            None,
            1.0,
            id="RecallTrainMean",
        ),
        param(
            "cv_binary_classification",
            RecallTestMean,
            "recall_mean",
            "Recall (macro) - MEAN",
            True,
            "test",
            None,
            0.93,
            id="RecallTestMean",
        ),
        param(
            "cv_binary_classification",
            RecallTrainStd,
            "recall_std",
            "Recall (macro) - STD",
            False,
            "train",
            None,
            0.0,
            id="RecallTrainStd",
        ),
        param(
            "cv_binary_classification",
            RecallTestStd,
            "recall_std",
            "Recall (macro) - STD",
            False,
            "test",
            None,
            0.04472135954999573,
            id="RecallTestStd",
        ),
    ),
)
def test_recall(
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
    monkeypatch.delattr(report.metrics.__class__, "recall")

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
