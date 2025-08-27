"""Class definition of the payload used to upload and send an artefact to ``hub``."""

from .artefact import CrossValidationReportArtefact, EstimatorReportArtefact

__all__ = [
    "EstimatorReportArtefact",
    "CrossValidationReportArtefact",
]
