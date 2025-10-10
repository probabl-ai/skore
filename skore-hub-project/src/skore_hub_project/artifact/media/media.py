"""Class definition of the payload used to associate a media with the report."""

from pydantic import Field

from skore_hub_project.artifact.artifact import Artifact
from skore_hub_project.protocol import CrossValidationReport, EstimatorReport


class Media(Artifact):
    """
    Payload used to associate a media with the report.

    Attributes
    ----------
    project : Project
        The project to which the artifact's payload must be associated.
    content_type : str
        The content-type of the artifact content.
    report : EstimatorReport | CrossValidationReport
        The report to pickled.
    name : str
        The name of the media.
    data_source : str | None
        The source of the data used to generate the media.
    """

    report: EstimatorReport | CrossValidationReport = Field(repr=False, exclude=True)
    name: str
    data_source: str | None = None
