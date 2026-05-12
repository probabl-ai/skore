import inspect
from functools import partialmethod
from hashlib import blake2b

from pydantic import ValidationError
from pytest import fixture, mark, param, raises

from skore._plugins.hub.artifact.media import (
    Coefficients,
    ImpurityDecrease,
    PermutationImportanceTest,
    PermutationImportanceTrain,
)
from skore._plugins.hub.artifact.upload import plan_upload
from skore._plugins.hub.json import dumps


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
            id="PermutationImportanceTest[estimator]",
        ),
        param(
            PermutationImportanceTest,
            "cv_binary_classification",
            "permutation_importance",
            "test",
            id="PermutationImportanceTest[cross-validation]",
        ),
        param(
            PermutationImportanceTrain,
            "binary_classification",
            "permutation_importance",
            "train",
            id="PermutationImportanceTrain[estimator]",
        ),
        param(
            PermutationImportanceTrain,
            "cv_binary_classification",
            "permutation_importance",
            "train",
            id="PermutationImportanceTrain[cross-validation]",
        ),
        param(
            ImpurityDecrease,
            "binary_classification",
            "impurity_decrease",
            None,
            id="ImpurityDecrease[estimator]",
        ),
        param(
            ImpurityDecrease,
            "cv_binary_classification",
            "impurity_decrease",
            None,
            id="ImpurityDecrease[cross-validation]",
        ),
        param(
            Coefficients,
            "multiclass_classification",
            "coefficients",
            None,
            id="Coefficients[estimator] - multiclass",
        ),
        param(
            Coefficients,
            "regression",
            "coefficients",
            None,
            id="Coefficients[estimator] - regression",
        ),
        param(
            Coefficients,
            "cv_regression",
            "coefficients",
            None,
            id="Coefficients[cross-validation] - regression",
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
    request,
    project,
):
    report = request.getfixturevalue(report)

    function = getattr(report.inspection, accessor)
    function_kwargs = {"data_source": data_source} if data_source else {}
    result = function(**function_kwargs)
    content = serialize(result)
    expected_checksum = f"blake2b-{blake2b(content).hexdigest()}"

    # available accessor: ``plan_upload`` produces content + checksum locally.
    plan = plan_upload(Media(project=project, report=report))

    assert plan is not None
    assert plan.content_type == "application/vnd.dataframe"
    assert plan.checksum == expected_checksum

    # unavailable accessor: ``plan_upload`` returns None.
    report.clear_cache()
    monkeypatch.delattr(report.inspection.__class__, accessor)

    assert plan_upload(Media(project=project, report=report)) is None

    # wrong type
    with raises(
        ValidationError,
        match=f"Input should be an instance of {report.__class__.__name__}",
    ):
        Media(project=project, report=None)
