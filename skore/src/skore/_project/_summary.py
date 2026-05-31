"""Class definition of the summary object, used in ``skore`` project."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from pandas import Categorical, DataFrame, Timestamp, isna, to_datetime
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from skore._utils.repr import ReprHTMLMixin
from skore._utils.repr.html_repr import render_template

if TYPE_CHECKING:
    from typing import Any, Literal

    from skore import ComparisonReport, CrossValidationReport, EstimatorReport


# Columns that are never shown to the user (e.g. constant within a project).
_HIDDEN_COLUMNS = ["ml_task"]

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


def _dtype_to_html_kind(column: str, dataframe: DataFrame) -> str:
    """Map a frame column to the HTML table sort kind."""
    if column == "id":
        return "text"
    dtype = dataframe.dtypes[column]
    if is_datetime64_any_dtype(dtype):
        return "date"
    if is_numeric_dtype(dtype):
        return "number"
    return "text"


class Summary(ReprHTMLMixin):
    """
    Metadata and metrics for all reports persisted in a project at a given moment.

    A summary object stores the metadata and metrics of the reports that have been
    persisted in a project as a :class:`pandas.DataFrame`, accessible through the
    :meth:`frame` method. It implements a custom HTML representation that displays the
    information as a table. The table provides a per-row selection and builds a query
    string that can be copied and passed to :meth:`query` to recover a subset of
    reports.

    Use :meth:`query` to filter the reports and :meth:`compare` to load the
    corresponding report objects from the project.

    See Also
    --------
    :func:`~skore.compare` :
        Compare selected reports side by side.
    :meth:`Project.summarize` :
        Create a summary from a project.
    """

    def __init__(self, dataframe: DataFrame, project: Any = None) -> None:
        if not dataframe.empty:
            dataframe["date"] = to_datetime(dataframe["date"], errors="coerce")
            dataframe["learner"] = Categorical(dataframe["learner"])
            for column in ("key", "dataset", "ml_task", "report_type"):
                dataframe[column] = dataframe[column].astype("string")
        self._summary = dataframe
        self.project = project

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

        metric_columns = frame.select_dtypes(include="number").columns
        kept_metrics = frame[metric_columns].dropna(axis="columns", how="all").columns
        return frame[
            [
                column
                for column in frame.columns
                if column not in _HIDDEN_COLUMNS
                and (column not in metric_columns or column in kept_metrics)
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

    def compare(
        self,
        *,
        return_as: Literal["list", "report"] = "list",
    ) -> list[EstimatorReport | CrossValidationReport] | ComparisonReport:
        """
        Return the reports referenced by the summary object from the project.

        Parameters
        ----------
        return_as : {"list", "report"}, default="list"
            In what form the reports should be returned.

        Returns
        -------
        reports : list of reports or ComparisonReport
            If ``return_as="list"``, a list of
            :class:`~skore.EstimatorReport` or
            :class:`~skore.CrossValidationReport` objects.
            If ``return_as="report"``, a :class:`~skore.ComparisonReport`.

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

        if return_as == "report":
            try:
                return ComparisonReport(reports)
            except ValueError as e:
                raise RuntimeError(
                    f"Bad condition: the report mode is only applicable when "
                    f"reports have the same dataset.\n"
                    f"Found '{self._summary['dataset'].unique()}'.\n"
                    f"Please query the summary to make your selection."
                ) from e
        return reports

    def plot(self) -> None:
        """Plot a visual summary of the reports referenced by this object.

        Raises
        ------
        NotImplementedError
            This method is not implemented yet.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return repr(self._summary)

    def _cell(self, column: str, value: object) -> dict[str, str]:
        """Build the display/sort parts of a single HTML table cell."""
        if value is None or isna(value):
            return {"display": "", "sort": ""}

        if isinstance(value, Timestamp):
            text = value.isoformat()
            return {"display": text, "sort": text}

        if isinstance(value, float):
            return {"display": f"{value:.6g}", "sort": repr(value)}

        text = str(value)
        return {"display": text, "sort": text.lower()}

    def _html_repr(self) -> str:
        """Show the HTML representation of the summary as a table."""
        container_id = f"skore-summary-{uuid.uuid4().hex[:8]}"

        columns: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        filters: list[dict[str, Any]] = []

        if not self._summary.empty:
            frame = self.frame()

            # Column order: id, then metadata/metrics, with ``date`` pushed to the
            # very end so the metrics sit right after the identifiers.
            ordered_columns = [column for column in frame.columns if column != "date"]
            if "date" in frame.columns:
                ordered_columns.append("date")
            data_columns = ["id", *ordered_columns]
            columns = [
                {
                    "key": column,
                    "label": _verbose_name(column),
                    "kind": _dtype_to_html_kind(column, self._summary),
                }
                for column in data_columns
            ]

            for field in ("report_type", "learner", "dataset"):
                options = [
                    {"value": str(value)} for value in sorted(frame[field].unique())
                ]
                filters.append(
                    {
                        "field": field,
                        "label": _verbose_name(field),
                        "options": options,
                    }
                )

            for id, (_, row) in zip(
                frame.index.get_level_values("id"),
                frame.iterrows(),
                strict=True,
            ):
                cells = [
                    self._cell(column, id if column == "id" else row[column])
                    for column in data_columns
                ]
                date = row["date"]
                date_value = (
                    ""
                    if isna(date)
                    else (
                        date.isoformat() if isinstance(date, Timestamp) else str(date)
                    )
                )
                rows.append(
                    {
                        "id": id,
                        "key": str(row["key"]),
                        "report_type": str(row["report_type"]),
                        "learner": str(row["learner"]),
                        "dataset": str(row["dataset"]),
                        "date": date_value,
                        "cells": cells,
                    }
                )

        return render_template(
            "project/summary.html.j2",
            {
                "container_id": container_id,
                "report_title": "Project summary",
                "columns": columns,
                "rows": rows,
                "filters": filters,
                "has_rows": bool(rows),
            },
        )
