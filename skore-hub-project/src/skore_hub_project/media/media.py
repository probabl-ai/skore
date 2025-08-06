from abc import ABC, abstractmethod
from typing import Any, Literal, ClassVar
from functools import cached_property

from pydantic import BaseModel, computed_field, Field


Report = Any


class Representation(BaseModel):
    media_type: str
    value: str


class Media(ABC, BaseModel):
    key: str
    verbose_name: str | None = None
    category: Literal["performance", "feature_importance", "model", "data"]
    attributes: dict | None = None
    parameters: dict | None = None

    @computed_field
    @property
    @abstractmethod
    def representation(self) -> Representation: ...


class EstimatorHtmlRepr(Media):
    report: Report = Field(repr=False, exclude=True)
    key: Literal["estimator_html_repr"] = "estimator_html_repr"
    category: Literal["model"] = "model"

    @computed_field
    @cached_property
    @abstractmethod
    def representation(self) -> Representation:
        import sklearn.utils

        return Representation(
            media_type="text/html",
            value=sklearn.utils.estimator_html_repr(self.report.estimator_),
        )


class TableReport(Media):
    key: Literal["table_report"] = "table_report"
    verbose_name: Literal["Table report"] = "Table report"
    category: Literal["data"] = "data"

    @computed_field
    @cached_property
    @abstractmethod
    def representation(self) -> Representation:
        function = self.report.data.analyze
        function_parameters = signature(function).parameters
        function_kwargs = {k: v for k, v in kwargs.items() if k in function_parameters}

        display = function(**function_kwargs)
        json = display._to_json()

        return {
            "attributes": kwargs,
            "parameters": {},
            "representation": {
                "media_type": "application/vnd.skrub.table-report.v1+json",
                "value": json,
            },
        }


class TableReportTest(Media):
    attributes: Literal[{"data_source": "test"}] = {"data_source": "test"}
