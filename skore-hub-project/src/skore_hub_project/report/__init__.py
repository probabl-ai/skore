"""Class definitions of the payloads used to send a report to ``hub``."""

from .cross_validation_report import CrossValidationReportPayload
from .estimator_report import EstimatorReportPayload

__all__ = [
    "CrossValidationReportPayload",
    "EstimatorReportPayload",
]
