from functools import partialmethod

from pydantic import ValidationError
from pytest import fixture, mark, param, raises

from skore_hub_project import Project
from skore_hub_project.artifact.media import (
    Coefficients,
    MeanDecreaseImpurity,
    PermutationTest,
    PermutationTrain,
)
from skore_hub_project.artifact.serializer import Serializer


def serialize(result) -> bytes:
    import orjson

    if hasattr(result, "frame"):
        result = result.frame()

    return orjson.dumps(
        result.fillna("NaN").to_dict(orient="tight"),
        option=(orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY),
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


@mark.usefixtures("monkeypatch_artifact_hub_client")
@mark.usefixtures("monkeypatch_upload_routes")
@mark.usefixtures("monkeypatch_upload_with_mock")
@mark.parametrize(
    "Media,report,accessor,data_source",
    (
        param(
            PermutationTest,
            "binary_classification",
            "permutation",
            "test",
            id="PermutationTest",
        ),
        param(
            PermutationTrain,
            "binary_classification",
            "permutation",
            "train",
            id="PermutationTrain",
        ),
        param(
            MeanDecreaseImpurity,
            "binary_classification",
            "mean_decrease_impurity",
            None,
            id="MeanDecreaseImpurity",
        ),
        param(
            Coefficients,
            "regression",
            "coefficients",
            None,
            id="Coefficients",
        ),
        param(
            Coefficients,
            "cv_regression",
            "coefficients",
            None,
            id="Coefficients",
        ),
    ),
)
def test_feature_importance(
    monkeypatch,
    Media,
    report,
    accessor,
    data_source,
    upload_mock,
    request,
):
    project = Project("myworkspace", "myname")
    report = request.getfixturevalue(report)

    function = getattr(report.feature_importance, accessor)
    function_kwargs = {"data_source": data_source} if data_source else {}
    result = function(**function_kwargs)
    content = serialize(result)

    with Serializer(content) as serializer:
        checksum = serializer.checksum

    # available accessor
    assert Media(project=project, report=report).model_dump() == {
        "content_type": "application/vnd.dataframe",
        "name": accessor,
        "data_source": data_source,
        "checksum": checksum,
    }

    # ensure `upload` is well called
    assert upload_mock.called
    assert not upload_mock.call_args.args
    assert upload_mock.call_args.kwargs == {
        "project": project,
        "content": content,
        "content_type": "application/vnd.dataframe",
    }

    # unavailable accessor
    report.clear_cache()
    monkeypatch.delattr(report.feature_importance.__class__, accessor)
    upload_mock.reset_mock()

    assert Media(project=project, report=report).model_dump() == {
        "content_type": "application/vnd.dataframe",
        "name": accessor,
        "data_source": data_source,
        "checksum": None,
    }

    # ensure `upload` is not called
    assert not upload_mock.called

    # wrong type
    with raises(
        ValidationError,
        match=f"Input should be an instance of {report.__class__.__name__}",
    ):
        Media(project=project, report=None)
