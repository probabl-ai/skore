"""Class definition of the payload used to send a model category media to ``hub``."""

from functools import cached_property
from typing import Literal

from pydantic import Field, computed_field

from skore_hub_project.protocol import CrossValidationReport, EstimatorReport

from .media import Media, Representation


class EstimatorHtmlRepr(Media):  # noqa: D101
    report: EstimatorReport | CrossValidationReport = Field(repr=False, exclude=True)
    key: str = "estimator_html_repr"
    verbose_name: str = "Estimator HTML representation"
    category: Literal["model"] = "model"

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def representation(self) -> Representation:  # noqa: D102
        import sklearn.utils

        return Representation(
            media_type="text/html",
            value=sklearn.utils.estimator_html_repr(self.report.estimator),
        )
