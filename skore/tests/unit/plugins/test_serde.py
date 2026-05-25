import pytest

from skore._plugins.serde import externalize, internalize


@pytest.fixture(
    params=[
        "estimator_reports_regression",
        "cross_validation_reports_regression",
    ],
    ids=["estimator", "cross-validation"],
)
def report(request):
    return request.getfixturevalue(request.param)[0]


def test_report_rebuilds_after_smart_serde(report) -> None:
    artifacts: dict[str, bytes] = {}
    state = externalize(report.get_state(), artifacts.__setitem__)

    restored_state = internalize(state, artifacts.__getitem__)

    restored_report = report.__class__.from_state(restored_state)
    # test restored_report works:
    restored_report.data.analyze()
    restored_report.metrics.summarize()
