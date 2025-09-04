from functools import partialmethod
from inspect import signature

from pandas import DataFrame
from pydantic import ValidationError
from pytest import fixture, mark, param, raises
from skore_hub_project.media import (
    Coefficients,
    MeanDecreaseImpurity,
    PermutationTest,
    PermutationTrain,
)


@fixture(autouse=True)
def monkeypatch_permutation(monkeypatch):
    import skore

    monkeypatch.setattr(
        "skore.EstimatorReport.feature_importance.permutation",
        partialmethod(
            skore.EstimatorReport.feature_importance.permutation,
            seed=42,
        ),
    )


@mark.parametrize(
    "Media,report,accessor,verbose_name,attributes",
    (
        param(
            PermutationTest,
            "binary_classification",
            "permutation",
            "Feature importance - Permutation",
            {"data_source": "test", "method": "permutation"},
            id="PermutationTest",
        ),
        param(
            PermutationTrain,
            "binary_classification",
            "permutation",
            "Feature importance - Permutation",
            {"data_source": "train", "method": "permutation"},
            id="PermutationTrain",
        ),
        param(
            MeanDecreaseImpurity,
            "binary_classification",
            "mean_decrease_impurity",
            "Feature importance - Mean Decrease Impurity (MDI)",
            {"method": "mean_decrease_impurity"},
            id="MeanDecreaseImpurity",
        ),
        param(
            Coefficients,
            "regression",
            "coefficients",
            "Feature importance - Coefficients",
            {"method": "coefficients"},
            id="Coefficients",
        ),
        param(
            Coefficients,
            "cv_regression",
            "coefficients",
            "Feature importance - Coefficients",
            {"method": "coefficients"},
            id="Coefficients",
        ),
    ),
)
def test_feature_importance(
    monkeypatch,
    Media,
    report,
    accessor,
    verbose_name,
    attributes,
    request,
):
    report = request.getfixturevalue(report)

    function = getattr(report.feature_importance, accessor)
    function_parameters = signature(function).parameters
    function_kwargs = {k: v for k, v in attributes.items() if k in function_parameters}

    result = function(**function_kwargs)

    if not isinstance(result, DataFrame):
        result = result.frame()

    serialized = result.fillna("NaN").to_dict(orient="tight")

    # available accessor
    assert Media(report=report).model_dump() == {
        "key": accessor,
        "verbose_name": verbose_name,
        "category": "feature_importance",
        "attributes": attributes,
        "parameters": {},
        "representation": {
            "media_type": "application/vnd.dataframe",
            "value": serialized,
        },
    }

    # unavailable accessor
    monkeypatch.delattr(report.feature_importance.__class__, accessor)

    assert Media(report=report).model_dump() == {
        "key": accessor,
        "verbose_name": verbose_name,
        "category": "feature_importance",
        "attributes": attributes,
        "parameters": {},
        "representation": None,
    }

    # wrong type
    with raises(ValidationError):
        Media(report=None)
