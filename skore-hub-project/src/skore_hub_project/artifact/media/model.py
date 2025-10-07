"""Class definition of the payload used to send a model category media to ``hub``."""

from typing import ClassVar, Literal

from skore_hub_project.artifact.media.media import Media
from skore_hub_project.artifact.serializer import TxtSerializer


class EstimatorHtmlRepr(Media):  # noqa: D101
    serializer: ClassVar[type[TxtSerializer]] = TxtSerializer
    name: Literal["estimator_html_repr"] = "estimator_html_repr"
    media_type: Literal["text/html"] = "text/html"

    def object_to_upload(self) -> str:
        import sklearn.utils

        return sklearn.utils.estimator_html_repr(self.report.estimator)
