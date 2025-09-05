"""Class definition of the payload used to send a data category media to ``hub``."""

from functools import cached_property
from inspect import signature
from typing import Literal

import numpy as np
from pydantic import Field, computed_field
from skore import EstimatorReport

from skore_hub_project import switch_mpl_backend

from .media import Media, Representation


def _to_native(obj):
    # If it's a dictionary, recurse on values
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}

    # If it's a list, recurse on each element
    elif isinstance(obj, list):
        return [_to_native(v) for v in obj]

    # If it's a tuple, recurse on each element
    elif isinstance(obj, tuple):
        return tuple(_to_native(v) for v in obj)

    # If it's a numpy array, convert to list (recursively handles inner scalars too)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    # If it's a numpy scalar (e.g., np.int32, np.float64), convert to native Python type
    elif isinstance(obj, np.generic):
        return obj.item()

    # Otherwise, leave it as is
    else:
        return obj


class TableReport(Media):  # noqa: D101
    report: EstimatorReport = Field(repr=False, exclude=True)
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
            to_remove = ["dataframe", "sample_table"]
            table_report_representation = {
                k: v
                for k, v in table_report_display.summary.items()
                if k not in to_remove
            }
            table_report_representation["extract"] = (
                table_report_display.summary["dataframe"]
                .head(3)
                .to_dict(orient="split")
            )

            return Representation(
                media_type="application/vnd.skrub.table-report.v1+json",
                value=_to_native(table_report_representation),
            )


class TableReportTrain(TableReport):  # noqa: D101
    attributes: dict = {"data_source": "train"}


class TableReportTest(TableReport):  # noqa: D101
    attributes: dict = {"data_source": "test"}
