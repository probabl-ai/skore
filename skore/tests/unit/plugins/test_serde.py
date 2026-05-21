import pytest

from skore import CrossValidationReport, EstimatorReport
from skore._plugins.serde import restore_data_in_state, split_data_from_state


@pytest.fixture(
    params=[
        "estimator_reports_regression",
        "cross_validation_reports_regression",
    ],
    ids=["estimator", "cross-validation"],
)
def report(request):
    return request.getfixturevalue(request.param)[0]


@pytest.mark.parametrize(
    ("report_cls", "report_fixture"),
    [
        (EstimatorReport, "estimator_reports_regression"),
        (CrossValidationReport, "cross_validation_reports_regression"),
    ],
    ids=["estimator", "cross-validation"],
)
def test_report_rebuilds_after_smart_serde(
    request,
    report_cls: type[EstimatorReport] | type[CrossValidationReport],
    report_fixture: str,
) -> None:
    report = request.getfixturevalue(report_fixture)[0]
    state = report.get_state()
    artifacts = dict(split_data_from_state(state))

    restored_state = restore_data_in_state(state, artifacts.__getitem__)

    restored_report = report_cls.from_state(restored_state)
    # test restored_report works:
    restored_report.data.analyze()
    restored_report.metrics.summarize()
