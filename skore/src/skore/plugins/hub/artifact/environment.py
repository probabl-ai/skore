"""Definition of the payload associating an environment snapshot with the report."""

from logging import getLogger
from platform import python_version
from typing import Literal

from skore.plugins.hub.artifact.artifact import Artifact
from skore.plugins.requirements import infer

logger = getLogger(__name__)


class Environment(Artifact):
    """
    Payload associating an environment snapshot with the report.

    Attributes
    ----------
    project : Project
        The project to which the artifact's payload must be associated.
    content_type : str
        The content-type of the artifact content.
    python_version : str
        The Python version from the environment snapshot used to create the report.

    Notes
    -----
    It uploads a lockfile-like list of package requirements inferred from that
    environment, in a lazy way. Returns ``None`` when requirements cannot be inferred.
    """

    content_type: Literal["text/plain"] = "text/plain"
    python_version: str = python_version()

    def content_to_upload(self) -> str | None:
        """
        Content of the environment snapshot used to create the report.

        Returns
        -------
        str or None
            A newline-separated list of ``name==version`` requirements from that
            environment, or ``None`` if requirements cannot be inferred.
        """
        try:
            return "\n".join(
                f"{requirement['name']}=={requirement['version']}"
                for requirement in infer()
            )
        except Exception:
            logger.debug("Failed to infer environment", exc_info=True)
            return None
