"""Class definition of the summary object, used in ``skore`` project."""

from __future__ import annotations

import math
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from pandas import Categorical, DataFrame, Index, MultiIndex, RangeIndex

from skore._utils.repr import ReprHTMLMixin
from skore._utils.repr.html_repr import render_template

if TYPE_CHECKING:
    from typing import Any, Literal

    from skore import ComparisonReport, CrossValidationReport, EstimatorReport


_METADATA_COLUMNS = ["key", "date", "learner", "dataset", "ml_task", "report_type"]
# Columns that are never shown to the user (e.g. constant within a project).
_HIDDEN_COLUMNS = ["ml_task"]
# Columns rendered with a middle ellipsis in the HTML table.
_ELLIPSIS_COLUMNS = ["id", "dataset"]

_COLUMN_LABELS = {
    "id": "ID",
    "key": "Key",
    "date": "Date",
    "learner": "Learner",
    "dataset": "Dataset",
    "report_type": "Report type",
    "rmse": "RMSE",
    "log_loss": "Log loss",
    "roc_auc": "ROC AUC",
    "fit_time": "Fit time (s)",
    "predict_time": "Predict time (s)",
}


def _verbose_name(column: str) -> str:
    """Return a human-readable name for ``column``."""
    if column in _COLUMN_LABELS:
        return _COLUMN_LABELS[column]
    for suffix, qualifier in (("_mean", "mean"), ("_std", "std")):
        if column.endswith(suffix):
            base = column[: -len(suffix)]
            return f"{_verbose_name(base)} ({qualifier})"
    return column.replace("_", " ").capitalize()


def _column_kind(column: str) -> str:
    """Return the sort kind (``number``, ``date`` or ``text``) for ``column``."""
    if column == "date":
        return "date"
    if column in _METADATA_COLUMNS or column == "id":
        return "text"
    return "number"


def _format_date(value: object) -> str:
    """Format an ISO date string, truncating the time to ``HH:MM:SS``."""
    try:
        return datetime.fromisoformat(str(value)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return str(value)


def _middle_ellipsis(value: str, head: int = 8, tail: int = 6) -> str:
    """Truncate the middle of ``value``, keeping its start and end."""
    if len(value) <= head + tail + 3:
        return value
    return f"{value[:head]}...{value[-tail:]}"


class Summary(ReprHTMLMixin):
    """
    Metadata and metrics for all reports persisted in a project at a given moment.

    A summary object stores the metadata and metrics of the reports that have been
    persisted in a project as a :class:`pandas.DataFrame`, accessible through the
    :meth:`frame` method. It implements a custom HTML representation that displays the
    information as a table. The table provides a per-row selection and builds a query
    string that can be copied and passed to :meth:`query` to recover a subset of
    reports.

    Use :meth:`query` to filter the reports and :meth:`reports` to load the
    corresponding report objects from the project.

    See Also
    --------
    :func:`~skore.compare` :
        Compare selected reports side by side.
    :meth:`Project.summarize` :
        Create a summary from a project.
    """

    def __init__(self, dataframe: DataFrame, project: Any = None) -> None:
        self._summary = dataframe
        self.project = project

    @staticmethod
    def factory(project, /) -> Summary:
        """
        Construct a summary object from ``project`` at a given moment.

        Parameters
        ----------
        project : ``skore._plugins.local.Project`` | ``skore._plugins.hub.Project``
            The project from which the summary object is to be constructed.

        Returns
        -------
        summary : Summary
            Metadata and metrics for every report persisted in ``project``.

        Notes
        -----
        This function is not intended for direct use. Instead simply use the accessor
        :meth:`skore.Project.summarize`.
        """
        summary = DataFrame(project.summarize(), copy=False)

        if not summary.empty:
            summary["learner"] = Categorical(summary["learner"])
            summary.index = MultiIndex.from_arrays(
                [
                    RangeIndex(len(summary)),
                    Index(summary.pop("id"), name="id", dtype=str),
                ]
            )

        return Summary(summary, project)

    def frame(
        self, *, report_type: Literal["estimator", "cross-validation"] | None = None
    ) -> DataFrame:
        """
        Return the metadata and metrics as a :class:`pandas.DataFrame`.

        Parameters
        ----------
        report_type : {"estimator", "cross-validation"}, default=None
            Filter the rows to a specific type of report. When ``None``, all reports
            are returned.

        Returns
        -------
        frame : pandas.DataFrame
            The metadata and metrics of the reports. Metric columns that are entirely
            empty (e.g. metrics not applicable to the selected report type or ML task)
            are dropped to only keep useful information. The ``ml_task`` column is
            dropped as well, as it is constant within a project.
        """
        frame = self._summary

        if report_type is not None:
            frame = frame[frame["report_type"] == report_type]

        metric_columns = [
            column for column in frame.columns if column not in _METADATA_COLUMNS
        ]
        kept_metrics = frame[metric_columns].dropna(axis="columns", how="all").columns
        return frame[
            [
                column
                for column in frame.columns
                if column not in _HIDDEN_COLUMNS
                and (column in _METADATA_COLUMNS or column in kept_metrics)
            ]
        ]

    def query(self, expr: str) -> Summary:
        """
        Filter the summary using a :meth:`pandas.DataFrame.query` expression.

        Parameters
        ----------
        expr : str
            The query string to evaluate. The report ``id`` is available as an index
            level, so selections built from the HTML representation (e.g.
            ``"id in ['ab12cd34']"``) can be used to recover specific reports.

        Returns
        -------
        summary : Summary
            A new summary restricted to the rows matching ``expr``.
        """
        return Summary(self._summary.query(expr), self.project)

    def reports(
        self,
        *,
        return_as: Literal["list", "comparison"] = "list",
    ) -> list[EstimatorReport | CrossValidationReport] | ComparisonReport:
        """
        Return the reports referenced by the summary object from the project.

        Parameters
        ----------
        return_as : {"list", "comparison"}, default="list"
            In what form the reports should be returned.

        Returns
        -------
        reports : list of reports or ComparisonReport
            If ``return_as="list"``, a list of
            :class:`~skore.EstimatorReport` or
            :class:`~skore.CrossValidationReport` objects.
            If ``return_as="comparison"``, a :class:`~skore.ComparisonReport`.

        See Also
        --------
        :func:`~skore.compare` :
            Compare reports side by side.
        """
        from skore import ComparisonReport

        if self._summary.empty:
            return []

        if self.project is None or "id" not in self._summary.index.names:
            raise RuntimeError("Bad condition: it is not a valid `Summary` object.")

        reports = [
            self.project.get(id) for id in self._summary.index.get_level_values("id")
        ]

        if return_as == "comparison":
            try:
                return ComparisonReport(reports)
            except ValueError as e:
                raise RuntimeError(
                    f"Bad condition: the comparison mode is only applicable when "
                    f"reports have the same dataset.\n"
                    f"Found '{self._summary['dataset'].unique()}'.\n"
                    f"Please query the summary to make your selection."
                ) from e
        return reports

    def __repr__(self) -> str:
        return repr(self._summary)

    def _cell(self, column: str, value: object) -> dict[str, str]:
        """Build the display/sort/title parts of a single HTML table cell."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return {"display": "", "sort": "", "title": ""}

        if column == "date":
            display = _format_date(value)
            return {"display": display, "sort": str(value), "title": ""}

        if isinstance(value, float):
            return {"display": f"{value:.6g}", "sort": repr(value), "title": ""}

        text = str(value)
        if column in _ELLIPSIS_COLUMNS:
            return {
                "display": _middle_ellipsis(text),
                "sort": text.lower(),
                "title": text,
            }
        return {"display": text, "sort": text.lower(), "title": ""}

    def _html_repr(self) -> str:
        """Show the HTML representation of the summary as a table."""
        container_id = f"skore-summary-{uuid.uuid4().hex[:8]}"

        columns: list[dict[str, str]] = []
        rows: list[dict[str, Any]] = []
        filters: list[dict[str, Any]] = []

        if not self._summary.empty:
            frame = self.frame()

            data_columns = ["id", *frame.columns]
            columns = [
                {
                    "key": column,
                    "label": _verbose_name(column),
                    "kind": _column_kind(column),
                }
                for column in data_columns
            ]

            # Each filterable column exposes its sorted unique values; long hashes
            # (e.g. ``dataset``) are middle-ellipsized for display but matched in full.
            for field in ("report_type", "learner", "dataset"):
                ellipsize = field in _ELLIPSIS_COLUMNS
                options = []
                for value in sorted(frame[field].unique()):
                    text = str(value)
                    options.append(
                        {
                            "value": text,
                            "label": _middle_ellipsis(text) if ellipsize else text,
                            "title": text if ellipsize else "",
                        }
                    )
                filters.append(
                    {"field": field, "label": _verbose_name(field), "options": options}
                )

            for id, (_, row) in zip(
                frame.index.get_level_values("id"),
                frame.iterrows(),
                strict=True,
            ):
                cells = [self._cell("id", id)]
                cells.extend(
                    self._cell(column, value)
                    for column, value in zip(frame.columns, row, strict=True)
                )
                date = row["date"]
                date_value = (
                    ""
                    if date is None or (isinstance(date, float) and math.isnan(date))
                    else str(date)
                )
                rows.append(
                    {
                        "id": id,
                        "report_type": str(row["report_type"]),
                        "learner": str(row["learner"]),
                        "dataset": str(row["dataset"]),
                        "date": date_value,
                        "cells": cells,
                    }
                )

        return render_template(
            "summary.html.j2",
            {
                "container_id": container_id,
                "report_title": "Project summary",
                "columns": columns,
                "rows": rows,
                "filters": filters,
                "has_rows": bool(rows),
            },
        )
