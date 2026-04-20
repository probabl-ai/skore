import pytest


@pytest.fixture(
    params=[
        "comparison_estimator_reports_binary_classification",
        "comparison_cross_validation_reports_binary_classification",
    ]
)
def report(request):
    return request.getfixturevalue(request.param)
