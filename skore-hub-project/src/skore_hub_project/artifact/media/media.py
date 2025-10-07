"""Class definition of the payload used to send a media to ``hub``."""

from abc import ABC

from pydantic import Field

from skore_hub_project.artifact.artifact import Artifact
from skore_hub_project.protocol import CrossValidationReport, EstimatorReport


class Media(Artifact, ABC):
    """
    Payload used to send a media to ``hub``.

    Attributes
    ----------
    """

    report: EstimatorReport | CrossValidationReport = Field(repr=False, exclude=True)
    name: str
    media_type: str
    data_source: str | None = None
