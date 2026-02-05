from functools import partialmethod

from pydantic import ValidationError
from pytest import fixture, mark, param, raises

from skore_hub_project.artifact.media import (
    Coefficients,
    ImpurityDecrease,
    PermutationImportanceTest,
    PermutationImportanceTrain,
)
from skore_hub_project.artifact.serializer import Serializer
from skore_hub_project.project.project import Project


def serialize(result) -> bytes:
    import orjson

    if hasattr(result, "frame"):
        result = result.frame()

    return orjson.dumps(
        result.fillna("NaN").to_dict(orient="tight"),
        option=(orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY),
    )


@fixture(autouse=True)
def monkeypatch_permutation_importance(monkeypatch):
    import skore

    monkeypatch.setattr(
        "skore.EstimatorReport.inspection.permutation_importance",
        partialmethod(
            skore.EstimatorReport.inspection.permutation_importance,
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
            PermutationImportanceTest,
            "binary_classification",
            "permutation_importance",
            "test",
            id="PermutationImportanceTest",
        ),
        param(
            PermutationImportanceTrain,
            "binary_classification",
            "permutation_importance",
            "train",
            id="PermutationImportanceTrain",
        ),
        param(
            ImpurityDecrease,
            "binary_classification",
            "impurity_decrease",
            None,
            id="ImpurityDecrease",
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
def test_inspection(
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

    function = getattr(report.inspection, accessor)
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
    monkeypatch.delattr(report.inspection.__class__, accessor)
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
