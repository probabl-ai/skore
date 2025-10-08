"""Class definition of the payload used to send a model category media to ``hub``."""

from typing import Literal

from skore_hub_project.artifact.media.media import Media


class EstimatorHtmlRepr(Media):  # noqa: D101
    name: Literal["estimator_html_repr"] = "estimator_html_repr"
    content_type: Literal["text/html"] = "text/html"

    def content_to_upload(self) -> str:
        import sklearn.utils

        return sklearn.utils.estimator_html_repr(self.report.estimator)
