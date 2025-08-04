import inspect

from skore import CrossValidationReport


def test_report_can_be_rebuilt_using_parameters(
    cross_validation_reports_binary_classification,
):
    report, _ = cross_validation_reports_binary_classification
    parameters = {}

    assert isinstance(report, CrossValidationReport)

    for parameter in inspect.signature(CrossValidationReport).parameters:
        assert hasattr(report, parameter), f"The parameter '{parameter}' must be stored"

        parameters[parameter] = getattr(report, parameter)

    CrossValidationReport(**parameters)
