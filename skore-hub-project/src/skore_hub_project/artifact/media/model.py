"""Definition of the payload used to associate a model category media with report."""

from typing import Literal

from sklearn.utils import estimator_html_repr

from skore_hub_project.artifact.media.media import Media, Report


class EstimatorHtmlRepr(Media[Report]):  # noqa: D101
    name: Literal["estimator_html_repr"] = "estimator_html_repr"
    data_source: None = None
    content_type: Literal["text/html"] = "text/html"

    def compute(self) -> None:  # noqa: D102
        if self.computed:
            return

        self.computed = True
        self.filepath.write_bytes(
            str.encode(
                estimator_html_repr(self.report.estimator),
                encoding="utf-8",
            )
        )
