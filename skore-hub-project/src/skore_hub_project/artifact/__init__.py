"""Class definition of the payload used to upload and send an artifact to ``hub``."""

from .artifact import CrossValidationReportArtifact, EstimatorReportArtifact

__all__ = [
    "EstimatorReportArtifact",
    "CrossValidationReportArtifact",
]
