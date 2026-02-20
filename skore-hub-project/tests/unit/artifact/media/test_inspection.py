import inspect
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
from skore_hub_project.json import dumps


def serialize(result) -> bytes:
    if hasattr(result, "frame"):
        # FIXME: in the future, all inspection methods should have an aggregate
        # parameter and we should be sending unaggregated data to the hub.
        if "aggregate" in inspect.signature(result.frame).parameters:
            result = result.frame(aggregate=None)
        else:
            result = result.frame()

    return dumps(
        result.astype(object).where(result.notna(), "NaN").to_dict(orient="tight")
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


@mark.filterwarnings(
    # ignore deprecation warning due to `scikit-learn` misusing `scipy` arguments,
    # raised by `scipy`
    (
        "ignore:scipy.optimize.*The `disp` and `iprint` options of the L-BFGS-B solver "
        "are deprecated:DeprecationWarning"
    ),
)
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
            "multiclass_classification",
            "coefficients",
            None,
            id="Coefficients",
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
@mark.respx()
def test_inspection(
    monkeypatch,
    Media,
    report,
    accessor,
    data_source,
    upload_mock,
    request,
    project,
):
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
