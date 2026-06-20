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


_IGNORED_COLUMNS = ["ml_task"]
_METADATA_COLUMNS = frozenset(
    {"key", "date", "learner", "dataset", "ml_task", "report_type"}
)
# Order of columns in the HTML table; metric columns are inserted at ``...``.
_COLUMN_ORDER = ("id", "key", ..., "date", "learner", "dataset", "report_type")
_HIDDEN_COLUMNS = ("learner", "dataset", "report_type")

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
            dataframe = self._coalesce_mean_metrics(dataframe)
        self._summary = dataframe
        self.project = project

    @staticmethod
    def _verbose_name(column: str) -> str:
        """Return a human-readable name for ``column``."""
        if column in _COLUMN_LABELS:
            return _COLUMN_LABELS[column]
        if column.endswith("_std"):
            base = column[: -len("_std")]
            return f"{Summary._verbose_name(base)} (std)"
        return column.replace("_", " ").capitalize()

    @staticmethod
    def _coalesce_mean_metrics(dataframe: DataFrame) -> DataFrame:
        """Merge ``metric_mean`` columns into the matching ``metric`` column."""
        for mean_column in [
            column for column in dataframe.columns if column.endswith("_mean")
        ]:
            base = mean_column[: -len("_mean")]
            if base not in dataframe.columns:
                dataframe = dataframe.rename(columns={mean_column: base})
                continue
            if not dataframe[mean_column].isna().all():
                if dataframe[base].isna().all():
                    dataframe[base] = dataframe[mean_column]
                else:
                    dataframe[base] = dataframe[base].combine_first(
                        dataframe[mean_column]
                    )
            dataframe = dataframe.drop(columns=mean_column)
        return dataframe

    @staticmethod
    def _column_role(column: str) -> str:
        """Return ``metadata``, ``metric``, or ``std`` for a frame column."""
        if column in _METADATA_COLUMNS:
            return "metadata"
        if column.endswith("_std"):
            return "std"
        return "metric"

    @staticmethod
    def _is_hidden_by_default(column: str) -> bool:
        """Return whether ``column`` should start hidden in the HTML table."""
        return column in _HIDDEN_COLUMNS or column.endswith("_std")

    def _dtype_to_html_kind(self, column: str) -> str:
        """Map a frame column to the HTML table sort kind."""
        if column == "id":
            return "text"
        dtype = self._summary.dtypes[column]
        if is_datetime64_any_dtype(dtype):
            return "date"
        if is_numeric_dtype(dtype):
            return "number"
        return "text"

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
            The metadata and metrics of the reports.
        """
        frame = self._summary

        if report_type is not None:
            frame = frame[frame["report_type"] == report_type]

        return frame[
            [
                column
                for column in frame.columns
                if column not in _IGNORED_COLUMNS
                and (column in _METADATA_COLUMNS or not frame[column].isna().all())
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

    @staticmethod
    def _cell(record: dict[str, Any], column: str) -> dict[str, str]:
        """Build the display/sort parts of a single HTML table cell."""
        value = record[column]
        if isna(value):
            cell = {"display": "", "sort": ""}
        elif isinstance(value, Timestamp):
            text = value.isoformat()
            cell = {"display": text, "sort": text}
        elif isinstance(value, float):
            cell = {"display": "", "sort": repr(float(value))}
        else:
            text = str(value)
            cell = {"display": text, "sort": text.lower()}
        std_column = f"{column}_std"
        if std_column in record:
            std_value = record[std_column]
            cell["std"] = (
                repr(float(std_value))
                if std_value is not None and not isna(std_value)
                else "nan"
            )
        return cell

    def _html_repr(self) -> str:
        """Show the HTML representation of the summary as a table."""
        container_id = f"skore-summary-{uuid.uuid4().hex[:8]}"

        if self._summary.empty:
            return render_template(
                "project/summary.html.j2",
                {
                    "container_id": container_id,
                    "report_title": "Project summary",
                    "columns": [],
                    "rows": [],
                    "filters": [],
                    "has_rows": False,
                },
            )

        frame = self.frame()

        known = {column for column in _COLUMN_ORDER if column is not ...}
        metric_columns = [column for column in frame.columns if column not in known]
        data_columns = []
        for column in _COLUMN_ORDER:
            if column is ...:
                data_columns.extend(metric_columns)
            elif column == "id" or column in frame.columns:
                data_columns.append(column)

        columns = [
            {
                "key": column,
                "label": self._verbose_name(column),
                "kind": self._dtype_to_html_kind(column),
                "role": self._column_role(column),
                "hidden_by_default": self._is_hidden_by_default(column),
            }
            for column in data_columns
        ]

        filters = [
            {
                "field": field,
                "label": self._verbose_name(field),
                "options": [
                    {"value": str(value)} for value in sorted(frame[field].unique())
                ],
            }
            for field in ("report_type", "learner", "dataset")
        ]

        rows = [
            {
                "id": record["id"],
                "key": str(record["key"]),
                "report_type": str(record["report_type"]),
                "learner": str(record["learner"]),
                "dataset": str(record["dataset"]),
                "date": self._cell(record, "date")["display"],
                "cells": [self._cell(record, column) for column in data_columns],
            }
            for record in frame.assign(id=frame.index.get_level_values("id")).to_dict(
                orient="records"
            )
        ]

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
