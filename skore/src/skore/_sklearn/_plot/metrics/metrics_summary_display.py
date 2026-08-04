from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Literal, NotRequired, TypedDict, cast

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.utils.validation import _is_arraylike

from skore._sklearn._plot.base import BOXPLOT_STYLE, DisplayMixin
from skore._sklearn.metrics import Metric
from skore._sklearn.types import (
    Aggregate,
    DataSource,
    PositiveLabel,
    ReportType,
)
from skore._utils._index import flatten_multi_index, squeeze_single_column

MetricIndexKey = Literal["metric", "label", "output", "average"]
MetricColumnKey = Literal["estimator", "data_source", "split"]

METRIC_INDEX_KEYS: tuple[MetricIndexKey, ...] = ("metric", "label", "output", "average")
METRIC_DIMENSION_KEYS = METRIC_INDEX_KEYS[1:]
PIVOT_VALUE_COLUMN = "score"
PIVOT_META_COLUMN = "greater_is_better"


class MetricsSummaryRow(TypedDict):
    """A single row rendered by ``MetricsSummaryDisplay``.

    Parameters
    ----------
    name : str
        Technical metric name (e.g. ``"accuracy"``); matches the key under which
        the metric is registered in :attr:`EstimatorReport._metric_registry`.
    verbose_name : str
        Human-readable metric name shown in the display.
    estimator : str
        Name shown in the display.
    data_source : {"train", "test"}
        Dataset split used to compute the metric.
    greater_is_better : bool or None
        Whether higher or lower values are better.
    score : Any
        Scalar metric value stored in the row.
    label : label, default=None
        Class label for per-class classification metrics.
    average : str, default=None
        Averaging mode when a metric is aggregated across labels or outputs.
    output : int, default=None
        Output index for multioutput regression metrics.
    split : int, optional
        Cross-validation split index.
    """

    name: str
    verbose_name: str
    estimator: str
    data_source: DataSource
    greater_is_better: bool | None
    score: Any
    label: PositiveLabel | None
    average: str | None
    output: int | None
    split: NotRequired[int]


class MetricsSummaryDisplay(DisplayMixin):
    """Summarize evaluation metrics in a table.

    Parameters
    ----------
    summary : pandas.DataFrame
        Long-format dataframe storing one row per metric observation, with the
        metric scores and their metadata (e.g. ``name``, ``verbose_name``,
        ``estimator``, ``data_source``, ``label``, ``output``, ``average``,
        ``split``, ``score``).

    report_type : {"estimator", "comparison-estimator", "cross-validation", \
            "comparison-cross-validation"}
        The type of report.

    errors : list of tuple of (Metric, Exception)
        Metric failures encountered while building the summary.

    Attributes
    ----------
    summary : pandas.DataFrame
        The long-format dataframe storing the metric scores and metadata.

    report_type : ReportType
        The type of report.

    See Also
    --------
    EstimatorReport.metrics.summarize : Create this display from a report.
    RocCurveDisplay : Plot ROC curves.
    PrecisionRecallCurveDisplay : Plot precision-recall curves.
    ConfusionMatrixDisplay : Display the confusion matrix.
    PredictionErrorDisplay : Plot regression prediction error.
    """

    _default_barplot_kwargs: dict[str, Any] = {
        "aspect": 2,
        "height": 6,
        "palette": "tab10",
    }
    _default_stripplot_kwargs: dict[str, Any] = {
        "alpha": 0.5,
        "aspect": 2,
        "height": 6,
        "palette": "tab10",
    }
    _default_boxplot_kwargs: dict[str, Any] = BOXPLOT_STYLE

    def __init__(
        self,
        summary: pd.DataFrame,
        report_type: ReportType,
        errors: list[tuple[Metric, Exception]],
    ):
        self.summary = summary
        self.report_type = report_type
        # Remove duplicates while preserving order
        # Use repr because Metrics and Exceptions are not comparable
        self.errors = list({repr(x): x for x in errors}.values())

    @classmethod
    def _compute_data_for_display(
        cls,
        rows: list[MetricsSummaryRow],
        *,
        report_type: ReportType,
        errors: list[tuple[Metric, Exception]],
    ) -> MetricsSummaryDisplay:
        """Build a display from metric rows, stored as a long-format DataFrame."""
        summary = pd.DataFrame(rows)

        if any(isinstance(r["label"], bool) for r in rows):
            summary["label"] = summary["label"].astype(pd.BooleanDtype())
        elif any(isinstance(r["label"], int) for r in rows):
            summary["label"] = summary["label"].astype(pd.Int64Dtype())

        if any(isinstance(r["output"], int) for r in rows):
            summary["output"] = summary["output"].astype(pd.Int64Dtype())

        if "average" in summary.columns:
            # ``multioutput`` can be an array-like value (e.g. raw_values); store
            # a stable string representation for grouping and display.
            summary["average"] = (
                summary["average"]
                .map(
                    lambda value: (
                        str(np.asarray(value).tolist())
                        if _is_arraylike(value)
                        else value
                    )
                )
                .astype("string")
            )

        return cls(summary, report_type=report_type, errors=errors)

    @staticmethod
    def _concatenate(
        child_displays: list[MetricsSummaryDisplay],
        *,
        report_type: ReportType,
        extra_rows_data: list[dict[str, Any]],
    ) -> MetricsSummaryDisplay:
        summary = pd.concat(
            [
                display.summary.assign(**extra_data)
                for display, extra_data in zip(
                    child_displays, extra_rows_data, strict=True
                )
            ],
            ignore_index=True,
        )
        errors = [error for display in child_displays for error in display.errors]
        return MetricsSummaryDisplay(summary, report_type=report_type, errors=errors)

    def _pivot_estimator(
        self,
        df: pd.DataFrame,
        index_cols: Sequence[str],
        column_cols: Sequence[str],
    ) -> pd.DataFrame:
        """Pivot a single-estimator table."""
        estimator = self.summary["estimator"].iloc[0]
        if not column_cols:
            # single data source and no column to spread across
            table = df.set_index(index_cols)[[PIVOT_VALUE_COLUMN]]
            table.columns = [estimator]
        else:
            table = df.pivot_table(
                index=index_cols,
                columns=column_cols,
                values=PIVOT_VALUE_COLUMN,
                aggfunc="first",
                sort=False,
            )
            table = table[["train", "test"]]
            table.columns = [f"{estimator} ({col})" for col in table.columns]
        return table

    def _pivot_cross_validation_single_source(
        self,
        df: pd.DataFrame,
        index_cols: Sequence[str],
        *,
        aggregate: Aggregate | None,
        estimator: str,
    ) -> pd.DataFrame:
        """Pivot one cross-validation source table."""
        if aggregate is None:
            table = df.pivot_table(
                index=index_cols,
                columns="split",
                values=PIVOT_VALUE_COLUMN,
                aggfunc="first",
                sort=False,
            )
            table.columns = pd.MultiIndex.from_tuples(
                [(estimator, f"Split #{col}") for col in table.columns]
            )
            table.columns.names = ["estimator", "split"]
        else:
            agg_list = [aggregate] if isinstance(aggregate, str) else list(aggregate)
            table = df.groupby(index_cols, dropna=False, sort=False)[
                PIVOT_VALUE_COLUMN
            ].agg(agg_list)
            table.columns = pd.MultiIndex.from_tuples(
                [(estimator, str(col)) for col in table.columns]
            )
            table.columns.names = ["estimator", "aggregate"]
        return table

    def _pivot_cross_validation(
        self,
        df: pd.DataFrame,
        index_cols: Sequence[str],
        *,
        aggregate: Aggregate | None,
    ) -> pd.DataFrame:
        """Pivot cross-validation metrics."""
        estimator = self.summary["estimator"].iloc[0]
        if "data_source" in df.columns:
            frames = []
            for data_source in ("train", "test"):
                source = df[df["data_source"] == data_source]
                source_frame = self._pivot_cross_validation_single_source(
                    source,
                    index_cols,
                    aggregate=aggregate,
                    estimator=estimator,
                )
                source_frame.columns = pd.MultiIndex.from_tuples(
                    [
                        (f"{col[0]} ({data_source})",) + col[1:]
                        for col in source_frame.columns
                    ]
                )
                frames.append(source_frame)
            return pd.concat(frames, axis="columns")
        return self._pivot_cross_validation_single_source(
            df,
            index_cols,
            aggregate=aggregate,
            estimator=estimator,
        )

    def _pivot_comparison_estimator(
        self,
        df: pd.DataFrame,
        index_cols: Sequence[str],
        column_cols: Sequence[str],
    ) -> pd.DataFrame:
        """Pivot comparison-estimator metrics."""
        table = df.pivot_table(
            index=index_cols,
            columns=column_cols,
            values=PIVOT_VALUE_COLUMN,
            aggfunc="first",
            sort=False,
        )
        if column_cols == ["estimator", "data_source"]:
            estimators = list(dict.fromkeys(df["estimator"]))
            table = table[
                [
                    (estimator, data_source)
                    for estimator in estimators
                    for data_source in ("train", "test")
                ]
            ]
            table.columns = [
                f"{estimator} ({data_source})"
                for estimator, data_source in table.columns
            ]
        else:
            table.columns.name = "estimator"
        return table

    def _pivot_comparison_cross_validation(
        self,
        df: pd.DataFrame,
        index_cols: Sequence[str],
        *,
        aggregate: Aggregate | None,
    ) -> pd.DataFrame:
        """Pivot comparison-cross-validation metrics."""
        if aggregate is None:
            table = df.pivot_table(
                index=index_cols,
                columns=["estimator", "split"],
                values=PIVOT_VALUE_COLUMN,
                aggfunc="first",
                sort=False,
            )
            table.columns = table.columns.set_levels(
                [f"Split #{level}" for level in table.columns.levels[1]],
                level=1,
            )
            table.columns.names = ["estimator", "split"]
        else:
            agg_list = [aggregate] if isinstance(aggregate, str) else list(aggregate)
            table = df.pivot_table(
                index=index_cols,
                columns=["estimator"],
                values=PIVOT_VALUE_COLUMN,
                aggfunc=agg_list,
                sort=False,
            )
            estimators = list(dict.fromkeys(df["estimator"]))
            table = table[
                [(str(agg), estimator) for agg in agg_list for estimator in estimators]
            ]
            table.columns.names = [None, "estimator"]
        return table

    def _finalize(
        self,
        table: pd.DataFrame,
        df: pd.DataFrame,
        index_cols: Sequence[str],
        *,
        favorability: bool,
        verbose_name: bool,
        flat_index: bool,
    ) -> pd.DataFrame | pd.Series:
        """Apply favorability, flat_index, and label-level cleanup to the table."""
        if favorability:
            favorability_col = (
                df.groupby(index_cols, dropna=False)[PIVOT_META_COLUMN]
                .first()
                .map({True: "(↗︎)", False: "(↘︎)"})
                .fillna("")
                .astype("string")
            )
            favorability_col.index = favorability_col.index.set_names(table.index.names)
            table["favorability"] = favorability_col

        if flat_index:
            if isinstance(table.columns, pd.MultiIndex):
                table.columns = flatten_multi_index(table.columns, lowercase=True)
            if isinstance(table.index, pd.MultiIndex):
                table.index = flatten_multi_index(table.index, lowercase=True)
        elif isinstance(table.index, pd.MultiIndex):
            levels = list(table.index.levels)
            for level_index, name in enumerate(table.index.names):
                if name == "label":
                    levels[level_index] = pd.Index(
                        [
                            "" if value == "" else str(value)
                            for value in levels[level_index]
                        ],
                        dtype="string",
                        name=name,
                    )
            table.index = table.index.set_levels(levels)

        if verbose_name:
            table.index.names = [
                None if name is None else name.replace("_", " ").title()
                for name in table.index.names
            ]
            table.columns.names = [
                None if name is None else name.replace("_", " ").title()
                for name in table.columns.names
            ]
            table = table.rename(columns={"favorability": "Favorability"})
        return squeeze_single_column(table, lowercase=not verbose_name)

    def frame(
        self,
        *,
        favorability: bool = False,
        verbose_name: bool = False,
        flat_index: bool = True,
        aggregate: Aggregate | None = ("mean", "std"),
    ) -> pd.DataFrame | pd.Series:
        """Return the metrics summary as a table.

        Parameters
        ----------
        favorability : bool, default=False
            Whether to add a column indicating whether higher ``(↗︎)`` or lower
            ``(↘︎)`` values are better for each metric.

        verbose_name : bool, default=False
            Whether to use the human-readable metric names instead of the
            technical names (e.g. ``"Accuracy"`` instead of ``"accuracy"``).
            Incompatible with ``flat_index=True``.

        flat_index : bool, default=True
            Whether to flatten MultiIndex row/column labels. Incompatible with
            ``verbose_name=True``.

        aggregate : {"mean", "std"}, list of such str or None, \
                default=("mean", "std")
            Only used for cross-validation reports. Functions to aggregate the
            scores across the cross-validation splits. ``None`` returns the
            scores for each split.

        Returns
        -------
        pandas.DataFrame or pandas.Series
            The metrics summary pivoted into a table. For layouts with a
            single value column, a :class:`pandas.Series` is returned.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> estimator = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(estimator, X, y)
        >>> metrics = report.metrics.summarize().frame()
        >>> metrics.loc["accuracy"]  # Series for single-estimator layout
        """
        if self.errors:
            warnings.warn(
                "\n".join(
                    f"Metric {metric.name!r} has failed: {error!r}"
                    for metric, error in self.errors
                ),
                stacklevel=2,
            )

        if verbose_name and flat_index:
            raise ValueError(
                "verbose_name=True is incompatible with flat_index=True. "
                "Use flat_index=False to preserve human-readable metric names, "
                "or set verbose_name=False."
            )

        summary = self.summary
        metric_col = "verbose_name" if verbose_name else "name"

        dimension_cols = [
            col for col in METRIC_DIMENSION_KEYS if summary[col].notna().any()
        ]
        has_both_sources = summary["data_source"].nunique() > 1

        columns: list[str] = []
        if "comparison" in self.report_type:
            columns.append("estimator")
        if "cross-validation" in self.report_type:
            columns.append("split")
        if has_both_sources:
            columns.append("data_source")
        columns.append(metric_col)
        columns.extend(dimension_cols)
        columns.extend(["score", "greater_is_better"])

        prepared = (
            summary[columns]
            .copy()
            .rename(columns={metric_col: "metric"})
            .reset_index(drop=True)
        )
        index_cols = ["metric", *dimension_cols]
        column_cols: list[MetricColumnKey]
        if self.report_type == "estimator":
            column_cols = ["data_source"] if has_both_sources else []
        elif self.report_type == "comparison-estimator":
            column_cols = (
                ["estimator", "data_source"] if has_both_sources else ["estimator"]
            )
        elif self.report_type == "cross-validation":
            column_cols = [] if aggregate is not None else ["split"]
        else:  # comparison-cross-validation
            column_cols = (
                ["estimator"] if aggregate is not None else ["estimator", "split"]
            )

        # Pivoting on a dimension column that contains NaN keys drops those rows, so
        # replace missing per-class/output/averaging values with an empty-string
        # sentinel (after resolving index keys, which relies on the NaN values).
        for col in METRIC_DIMENSION_KEYS:
            if col in prepared.columns and prepared[col].isna().any():
                prepared[col] = prepared[col].astype(object)
                prepared.loc[prepared[col].isna(), col] = ""

        if self.report_type == "estimator":
            table = self._pivot_estimator(prepared, index_cols, column_cols)
        elif self.report_type == "cross-validation":
            table = self._pivot_cross_validation(
                prepared,
                index_cols,
                aggregate=aggregate,
            )
        elif self.report_type == "comparison-estimator":
            table = self._pivot_comparison_estimator(prepared, index_cols, column_cols)
        else:
            table = self._pivot_comparison_cross_validation(
                prepared,
                index_cols,
                aggregate=aggregate,
            )

        return self._finalize(
            table,
            prepared,
            index_cols,
            favorability=favorability,
            verbose_name=verbose_name,
            flat_index=flat_index,
        )

    def _repr_html_(self) -> str:
        aggregate = cast(Aggregate, ("mean", "std"))
        frame = self.frame(
            aggregate=aggregate,
            verbose_name=True,
            flat_index=False,
        )
        html = (
            frame.to_frame()._repr_html_()
            if isinstance(frame, pd.Series)
            else frame._repr_html_()
        )
        lines = [
            html,
            (
                '<p role="note">Use <code>.frame()</code> to control the format'
                " of the output.</p>"
            ),
        ]
        lines.extend(
            f'<p role="note">Metric {metric.name!r} has failed: {error!r}</p>'
            for metric, error in self.errors
        )
        return "".join(lines)

    def __repr__(self) -> str:
        aggregate = cast(Aggregate, ("mean", "std"))
        frame = self.frame(
            aggregate=aggregate,
            verbose_name=False,
            flat_index=True,
        )
        lines = [
            f"{frame!r}",
            "Use .frame() to control the format of the output.",
        ]
        lines.extend(
            f"Metric {metric.name!r} has failed: {error!r}"
            for metric, error in self.errors
        )
        return "\n".join(lines)

    def available(self) -> list[str]:
        """List available metric names in the summary.

        Returns
        -------
        list of str
            Unique metric names from the ``name`` column of :attr:`summary`,
            in the order they first appear. These are the same names accepted
            by :meth:`plot`.
        """
        if self.summary.empty or "name" not in self.summary.columns:
            return []
        return list(dict.fromkeys(self.summary["name"].tolist()))

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        metric: str,
        subplot_by: Literal["auto", "estimator", "label", "output", "data_source"]
        | None = "auto",
    ) -> Figure:
        """Plot a single metric.

        Parameters
        ----------
        metric : str
            The metric to plot. Must be one of the names returned by
            :meth:`available` (the ``name`` column of :attr:`summary`).

        subplot_by : {"auto", "estimator", "label", "output", "data_source"} \
                or None, default="auto"
            The column to use for subplotting. If ``"auto"``, subplotting is
            performed only when comparing estimators in a multiclass
            classification or multi-output regression problem.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the metrics plot.
        """
        if self.errors:
            warnings.warn(
                "\n".join(
                    f"Metric {failed_metric.name!r} has failed: {error!r}"
                    for failed_metric, error in self.errors
                ),
                stacklevel=2,
            )
        return self._plot(metric=metric, subplot_by=subplot_by)

    def _plot_matplotlib(
        self,
        *,
        metric: str,
        subplot_by: Literal["auto", "estimator", "label", "output", "data_source"]
        | None = "auto",
    ) -> Figure:
        """Dispatch the plotting function for matplotlib backend."""
        frame = self._prepare_plot_frame(metric)

        barplot_kwargs = self._default_barplot_kwargs.copy()
        boxplot_kwargs = self._default_boxplot_kwargs.copy()
        stripplot_kwargs = self._default_stripplot_kwargs.copy()

        if "comparison" in self.report_type:
            return self._plot_comparison(
                frame=frame,
                report_type=self.report_type,
                subplot_by=subplot_by,
                barplot_kwargs=barplot_kwargs,
                boxplot_kwargs=boxplot_kwargs,
                stripplot_kwargs=stripplot_kwargs,
            )

        estimator_name = self.summary["estimator"].iloc[0]
        return self._plot_single_estimator(
            frame=frame,
            estimator_name=estimator_name,
            report_type=self.report_type,
            subplot_by=subplot_by,
            barplot_kwargs=barplot_kwargs,
            boxplot_kwargs=boxplot_kwargs,
            stripplot_kwargs=stripplot_kwargs,
        )

    def _prepare_plot_frame(self, metric: str) -> pd.DataFrame:
        """Filter and reshape the summary into a long frame for plotting."""
        available = self.available()
        if metric not in available:
            raise ValueError(
                f"Unknown metric: {metric!r}. Available metrics: {available!r}."
            )

        frame = self.summary.loc[self.summary["name"] == metric].copy()

        for col in ["label", "output", "average"]:
            if col in frame.columns and frame[col].isna().all():
                frame = frame.drop(columns=col)

        if "data_source" in frame.columns and frame["data_source"].nunique() == 1:
            frame = frame.drop(columns="data_source")

        if (
            "estimator" in frame.columns
            and frame["estimator"].nunique() == 1
            and "comparison" not in self.report_type
        ):
            frame = frame.drop(columns="estimator")

        if self.report_type in ("estimator", "comparison-estimator") and "split" in (
            frame.columns
        ):
            frame = frame.drop(columns="split")

        return frame

    @staticmethod
    def _get_columns_to_groupby(*, frame: pd.DataFrame) -> list[str]:
        """Get the available columns from which to group by."""
        columns_to_groupby = list[str]()
        for column in ("estimator", "data_source", "label", "output"):
            if column in frame.columns:
                columns_to_groupby.append(column)
        return columns_to_groupby

    @staticmethod
    def _decorate_matplotlib_axis(
        *,
        ax: Any,
        xlabel: str,
        ylabel: str = "",
    ) -> None:
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.axhspan(-0.5, 0.5, color="lightgray", alpha=0.4, zorder=0)

    def _categorical_plot(
        self,
        *,
        frame: pd.DataFrame,
        report_type: ReportType,
        hue: str | None = None,
        col: str | None = None,
        barplot_kwargs: dict[str, Any] | None = None,
        boxplot_kwargs: dict[str, Any] | None = None,
        stripplot_kwargs: dict[str, Any] | None = None,
    ) -> Figure:
        if "estimator" in report_type:
            facet = sns.catplot(
                data=frame,
                x="score",
                y="verbose_name",
                hue=hue,
                col=col,
                kind="bar",
                **(barplot_kwargs or {}),
            )
        else:
            facet = sns.catplot(
                data=frame,
                x="score",
                y="verbose_name",
                hue=hue,
                col=col,
                kind="strip",
                dodge=True,
                **(stripplot_kwargs or {}),
            ).map_dataframe(
                sns.boxplot,
                x="score",
                y="verbose_name",
                hue=hue,
                palette="tab10" if hue is not None else None,
                dodge=True,
                **(boxplot_kwargs or {}),
            )

        add_background_metric = hue is not None
        figure = facet.figure
        ax_grid = facet.axes.squeeze()
        xlabel = frame["verbose_name"].iloc[0]
        for ax in ax_grid.flatten():
            self._decorate_matplotlib_axis(ax=ax, xlabel=xlabel)
            if not add_background_metric:
                for patch in ax.patches:
                    patch.set_facecolor("lightgray")
                    patch.set_alpha(0.4)

        return figure

    def _plot_single_estimator(
        self,
        *,
        frame: pd.DataFrame,
        estimator_name: str,
        report_type: ReportType,
        subplot_by: Literal["auto", "estimator", "label", "output", "data_source"]
        | None,
        barplot_kwargs: dict[str, Any],
        boxplot_kwargs: dict[str, Any],
        stripplot_kwargs: dict[str, Any],
    ) -> Figure:
        """Plot metrics for an `EstimatorReport` or a `CrossValidationReport`."""
        columns_to_groupby = self._get_columns_to_groupby(frame=frame)
        if subplot_by == "auto":
            subplot_by = None

        if subplot_by is not None and not len(columns_to_groupby):
            raise ValueError(
                "No columns to group by. `subplot_by` is expected to be None or 'auto'."
            )
        if subplot_by is not None and subplot_by not in columns_to_groupby:
            raise ValueError(
                f"Column {subplot_by} not found in the frame. It should be one "
                f"of {', '.join(columns_to_groupby + ['auto', 'None'])}."
            )

        if subplot_by is None:
            hue = None if not len(columns_to_groupby) else columns_to_groupby[0]
            if hue is None:
                barplot_kwargs.pop("palette", None)
                stripplot_kwargs.pop("palette", None)
            col = None
        else:
            hue, col = None, subplot_by
            barplot_kwargs.pop("palette", None)
            stripplot_kwargs.pop("palette", None)

        figure = self._categorical_plot(
            frame=frame,
            report_type=report_type,
            hue=hue,
            col=col,
            barplot_kwargs=barplot_kwargs,
            boxplot_kwargs=boxplot_kwargs,
            stripplot_kwargs=stripplot_kwargs,
        )

        title = f"Metrics of {estimator_name}"
        if subplot_by is not None:
            title += f" by {subplot_by}"
        figure.suptitle(title)
        return figure

    def _plot_comparison(
        self,
        *,
        frame: pd.DataFrame,
        report_type: ReportType,
        subplot_by: Literal["auto", "estimator", "label", "output", "data_source"]
        | None,
        barplot_kwargs: dict[str, Any],
        boxplot_kwargs: dict[str, Any],
        stripplot_kwargs: dict[str, Any],
    ) -> Figure:
        """Plot metrics for a `ComparisonReport`."""
        hue: str | None = None
        columns_to_groupby = self._get_columns_to_groupby(frame=frame)

        if subplot_by not in ("auto", None) and subplot_by not in columns_to_groupby:
            additional_subplot_by = ["auto"]
            if "label" not in frame.columns and "output" not in frame.columns:
                additional_subplot_by.append("None")

            raise ValueError(
                f"Column {subplot_by} not found in the frame. It should be one "
                f"of {', '.join(columns_to_groupby + additional_subplot_by)}."
            )
        if subplot_by is None:
            if "label" in frame.columns:
                n_unique = frame["label"].nunique()
            elif "output" in frame.columns:
                n_unique = frame["output"].nunique()
            else:
                n_unique = 1
            if n_unique > 1:
                raise ValueError(
                    "There are multiple labels or outputs and `subplot_by` is `None`. "
                    "There is too much information to display on a single plot. "
                    "Please provide a column to group by using `subplot_by`."
                )

        if (frame.columns.isin(["label", "output"]).any() and subplot_by == "auto") or (
            subplot_by == "auto"
            and "estimator" in frame.columns
            and frame["estimator"].nunique() > 1
            and ("label" in frame.columns or "output" in frame.columns)
        ):
            subplot_by = "estimator"
        elif subplot_by == "auto":
            subplot_by = None

        if subplot_by is None:
            hue, col = columns_to_groupby[0], None
        else:
            hue_groupby = [
                column for column in columns_to_groupby if column != subplot_by
            ]
            hue = hue_groupby[0] if len(hue_groupby) else None
            col = subplot_by

            if hue is None:
                barplot_kwargs.pop("palette", None)
                stripplot_kwargs.pop("palette", None)

        figure = self._categorical_plot(
            frame=frame,
            report_type=report_type,
            hue=hue,
            col=col,
            barplot_kwargs={"sharey": True} | barplot_kwargs,
            boxplot_kwargs=boxplot_kwargs,
            stripplot_kwargs={"sharey": True} | stripplot_kwargs,
        )

        title = "Metrics"
        if subplot_by is not None:
            title += f" by {subplot_by}"
        figure.suptitle(title)
        return figure

    def set_style(  # type: ignore[override]
        self,
        *,
        policy: Literal["override", "update"] = "update",
        barplot_kwargs: dict[str, Any] | None = None,
        boxplot_kwargs: dict[str, Any] | None = None,
        stripplot_kwargs: dict[str, Any] | None = None,
    ):
        """Set the style parameters for the display.

        Parameters
        ----------
        policy : {"override", "update"}, default="update"
            Policy to use when setting the style parameters.

        barplot_kwargs : dict, default=None
            Keyword arguments passed to :func:`seaborn.barplot`.

        boxplot_kwargs : dict, default=None
            Keyword arguments passed to :func:`seaborn.boxplot`.

        stripplot_kwargs : dict, default=None
            Keyword arguments passed to :func:`seaborn.stripplot`.
        """
        return super().set_style(
            policy=policy,
            barplot_kwargs=barplot_kwargs or {},
            boxplot_kwargs=boxplot_kwargs or {},
            stripplot_kwargs=stripplot_kwargs or {},
        )
