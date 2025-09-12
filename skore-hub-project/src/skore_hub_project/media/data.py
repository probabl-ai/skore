"""Class definition of the payload used to send a data category media to ``hub``."""

from functools import cached_property
from inspect import signature
from typing import Literal

import numpy as np
from pydantic import Field, computed_field

from skore_hub_project import switch_mpl_backend
from skore_hub_project.protocol import CrossValidationReport, EstimatorReport

from .media import Media, Representation


def _to_native(obj):
    """Walk an object and cast all numpy types to native type.

    Useful for json serialization.
    """
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_native(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_to_native(v) for v in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


class TableReport(Media):  # noqa: D101
    report: EstimatorReport | CrossValidationReport = Field(repr=False, exclude=True)
    key: str = "table_report"
    verbose_name: str = "Table report"
    category: Literal["data"] = "data"

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def representation(self) -> Representation:  # noqa: D102
        with switch_mpl_backend():
            function = self.report.data.analyze
            function_parameters = signature(function).parameters
            function_kwargs = {
                k: v for k, v in self.attributes.items() if k in function_parameters
            }

            table_report_display = function(**function_kwargs)
            table_report = table_report_display.summary

            table_report["extract_head"] = (
                table_report["dataframe"].head(3).to_dict(orient="split")
            )

            table_report["extract_tail"] = (
                table_report["dataframe"].tail(3).to_dict(orient="split")
            )

            del table_report["dataframe"]
            del table_report["sample_table"]

            return Representation(
                media_type="application/vnd.skrub.table-report.v1+json",
                value=_to_native(table_report),
            )


class TableReportTrain(TableReport):  # noqa: D101
    attributes: dict = {"data_source": "train"}


class TableReportTest(TableReport):  # noqa: D101
    attributes: dict = {"data_source": "test"}
