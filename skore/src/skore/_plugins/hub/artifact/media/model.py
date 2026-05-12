"""Definition of the payload used to associate a model category media with report."""

from typing import Literal

from sklearn.utils import estimator_html_repr

from skore._plugins.hub.artifact.media.media import Media, Report


class EstimatorHtmlRepr(Media[Report]):  # noqa: D101
    name: Literal["estimator_html_repr"] = "estimator_html_repr"
    data_source: None = None
    content_type: Literal["text/html"] = "text/html"

    def content_to_upload(self) -> str:  # noqa: D102
        html: str = estimator_html_repr(self.report.estimator_)
        return html
