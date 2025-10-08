from pytest import mark, param
from skore_hub_project.protocol import CrossValidationReport, EstimatorReport


@mark.parametrize(
    "report,protocol,assertion",
    (
        param(
            "regression",
            EstimatorReport,
            True,
            id="EstimatorReport is EstimatorReport",
        ),
        param(
            "regression",
            CrossValidationReport,
            False,
            id="EstimatorReport is not CrossValidationReport",
        ),
        param(
            "cv_regression",
            CrossValidationReport,
            True,
            id="CrossValidationReport is CrossValidationReport",
        ),
        param(
            "cv_regression",
            EstimatorReport,
            False,
            id="CrossValidationReport is not EstimatorReport",
        ),
    ),
)
def test_validity(report, protocol, assertion, request):
    assert isinstance(request.getfixturevalue(report), protocol) is assertion
