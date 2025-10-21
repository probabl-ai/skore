"""Definition of the payload used to associate a model category media with report."""

from typing import Literal

from skore_hub_project.artifact.media.media import Media


class EstimatorHtmlRepr(Media):  # noqa: D101
    name: Literal["estimator_html_repr"] = "estimator_html_repr"
    data_source: None = None
    content_type: Literal["text/html"] = "text/html"

    def content_to_upload(self) -> str:  # noqa: D102
        import sklearn.utils

        return sklearn.utils.estimator_html_repr(self.report.estimator)
