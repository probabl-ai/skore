from __future__ import annotations

from numpy.testing import assert_almost_equal
from pydantic import ValidationError
from pytest import mark, param, raises
from skore_hub_project.metric import (
    BrierScoreTest,
    BrierScoreTestMean,
    BrierScoreTestStd,
    BrierScoreTrain,
    BrierScoreTrainMean,
    BrierScoreTrainStd,
)


@mark.parametrize(
    "report,Metric,name,verbose_name,greater_is_better,data_source,position,value",
    (
        param(
            "binary_classification",
            BrierScoreTrain,
            "brier_score",
            "Brier score",
            False,
            "train",
            None,
            0.007277500000000001,
            id="BrierScoreTrain",
        ),
        param(
            "binary_classification",
            BrierScoreTest,
            "brier_score",
            "Brier score",
            False,
            "test",
            None,
            0.09025999999999999,
            id="BrierScoreTest",
        ),
        param(
            "cv_binary_classification",
            BrierScoreTrainMean,
            "brier_score_mean",
            "Brier score - MEAN",
            False,
            "train",
            None,
            0.008439499999999999,
            id="BrierScoreTrainMean",
        ),
        param(
            "cv_binary_classification",
            BrierScoreTestMean,
            "brier_score_mean",
            "Brier score - MEAN",
            False,
            "test",
            None,
            0.060865999999999996,
            id="BrierScoreTestMean",
        ),
        param(
            "cv_binary_classification",
            BrierScoreTrainStd,
            "brier_score_std",
            "Brier score - STD",
            False,
            "train",
            None,
            0.0004868365678027891,
            id="BrierScoreTrainStd",
        ),
        param(
            "cv_binary_classification",
            BrierScoreTestStd,
            "brier_score_std",
            "Brier score - STD",
            False,
            "test",
            None,
            0.015918303694175455,
            id="BrierScoreTestStd",
        ),
    ),
)
def test_brier_score(
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
    monkeypatch.delattr(report.metrics.__class__, "brier_score")

    assert Metric(report=report).model_dump() == {
        "name": name,
        "verbose_name": verbose_name,
        "greater_is_better": greater_is_better,
        "data_source": data_source,
        "position": position,
        "value": None,
    }

    # wrong type
    with raises(
        ValidationError,
        match=f"Input should be an instance of {report.__class__.__name__}",
    ):
        Metric(report=None)
