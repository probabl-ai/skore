from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict, cast

import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from skore._sklearn._plot.base import BOXPLOT_STYLE, DisplayMixin
from skore._sklearn.types import (
    Aggregate,
    DataSource,
    PositiveLabel,
    ReportType,
)
from skore._utils._index import flatten_multi_index


class MetricsSummaryRow(TypedDict):
    """A single row rendered by ``MetricsSummaryDisplay``.

    Parameters
    ----------
    metric_verbose_name : str
        Human-readable metric name shown in the display.
    estimator_name : str
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

    metric_name: str
    metric_verbose_name: str
    estimator_name: str
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
    rows : list of MetricsSummaryRow
        The rows to display.

    report_type : {"estimator", "comparison-estimator", "cross-validation", \
            "comparison-cross-validation"}
        The type of report.

    Attributes
    ----------
    rows : list of MetricsSummaryRow
        Metric scores and metadata for each row of the summary.
    report_type : ReportType
        The type of report.
    data : pandas.DataFrame
        Rows as a DataFrame (read-only property).

    See Also
    --------
    EstimatorReport.metrics.summarize : Create this display from a report.
    RocCurveDisplay : Plot ROC curves.
    PrecisionRecallCurveDisplay : Plot precision-recall curves.
    ConfusionMatrixDisplay : Display the confusion matrix.
    PredictionErrorDisplay : Plot regression prediction error.

    Notes
    -----
    For cross-validation and comparison reports, :meth:`frame` can aggregate
    scores across splits or estimators using the ``aggregate`` parameter.
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
    _default_boxplot_kwargs: dict[str, Any] = {
        "whis": 1e10,
        **BOXPLOT_STYLE,
    }

    def __init__(
        self,
        rows: list[MetricsSummaryRow],
        report_type: ReportType,
    ):
        self.rows = rows
        self.report_type = report_type

    @property
    def data(self):
        """Return rows as a DataFrame, preserving nullable dtypes."""
        data = pd.DataFrame(self.rows)

        if any(isinstance(r["label"], bool) for r in self.rows):
            data["label"] = data["label"].astype(pd.BooleanDtype())
        elif any(isinstance(r["label"], int) for r in self.rows):
            data["label"] = data["label"].astype(pd.Int64Dtype())

        if any(isinstance(r["output"], int) for r in self.rows):
            data["output"] = data["output"].astype(pd.Int64Dtype())

        return data

    @staticmethod
    def _concatenate(
        child_displays: list[MetricsSummaryDisplay],
        *,
        report_type: ReportType,
        extra_rows_data: list[dict[str, Any]],
    ) -> MetricsSummaryDisplay:
        rows = []
        for display, extra_data in zip(child_displays, extra_rows_data, strict=True):
            rows.extend(
                [cast(MetricsSummaryRow, row | extra_data) for row in display.rows]
            )

        return MetricsSummaryDisplay(rows, report_type=report_type)

    @staticmethod
    def _flatten_index(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = flatten_multi_index(df.columns)
        if isinstance(df.index, pd.MultiIndex):
            df.index = flatten_multi_index(df.index)
        if isinstance(df.index, pd.Index):
            df.index = df.index.str.replace(r"\((.*)\)$", r"\1", regex=True)

        return df

    @staticmethod
    def _frame_estimator(
        data: pd.DataFrame,
        *,
        favorability: bool = False,
        flat_index: bool = False,
    ) -> pd.DataFrame:
        """Process estimator report data into a formatted dataframe."""
        df = data.copy()
        df = df.dropna(axis="columns", how="all")
        df = df.drop(columns="metric_name", errors="ignore")

        for col in df.columns.intersection(["label", "output", "average"]):
            df[col] = df[col].astype("string").fillna("")

        estimator_name = df.pop("estimator_name").iloc[0]
        index = df.columns.intersection(
            ["metric_verbose_name", "label", "output", "average"]
        ).to_list()
        df = df.set_index(index)

        # Rename columns as well as index names
        new_columns = {
            "metric_verbose_name": "Metric",
            "label": "Label",
            "output": "Output",
            "average": "Average",
            "score": estimator_name,
        }
        df = df.rename(columns=new_columns)

        if favorability:
            df["Favorability"] = (
                df["greater_is_better"]
                .map({True: "(↗︎)", False: "(↘︎)"})
                .fillna("")
                .astype("string")
            )
        df = df.drop(columns="greater_is_better")

        df.index = df.index.set_names(
            [new_columns.get(name, name) for name in df.index.names]
        )

        if df["data_source"].nunique() == 1:
            df = df.drop(columns="data_source")
        else:
            # Show metrics one column per data source
            df_pivoted = df.reset_index().pivot_table(
                index=df.index.names,
                columns="data_source",
                values=estimator_name,
                sort=False,
            )
            df_pivoted.columns = [
                f"{estimator_name} ({col})" for col in df_pivoted.columns
            ]

            if favorability:
                df_pivoted["Favorability"] = df.loc[
                    df["data_source"] == "test", "Favorability"
                ]

            df = df_pivoted.copy()

        if flat_index:
            df = MetricsSummaryDisplay._flatten_index(df)

        return df

    @staticmethod
    def _frame_cross_validation(
        data: pd.DataFrame,
        *,
        aggregate: Aggregate | None = ("mean", "std"),
        favorability: bool = False,
        flat_index: bool = False,
    ) -> pd.DataFrame:
        """Process cross-validation report data into a formatted dataframe."""
        df = data.copy()

        if df["data_source"].nunique() > 1:
            grouped = list(df.groupby("data_source", sort=False))
            frames = []
            favorability_col = None

            for data_source, source_df in grouped:
                source_frame = MetricsSummaryDisplay._frame_cross_validation(
                    source_df,
                    aggregate=aggregate,
                    favorability=True,
                    flat_index=False,
                )
                if favorability_col is None and "Favorability" in source_frame.columns:
                    favorability_col = source_frame.pop("Favorability")
                else:
                    source_frame = source_frame.drop(
                        columns="Favorability", errors="ignore"
                    )
                if isinstance(source_frame.columns, pd.MultiIndex):
                    source_frame.columns = pd.MultiIndex.from_tuples(
                        [
                            (f"{col[0]} ({data_source})",) + col[1:]
                            for col in source_frame.columns
                        ]
                    )
                else:
                    source_frame.columns = [
                        f"{col} ({data_source})" for col in source_frame.columns
                    ]
                frames.append(source_frame)

            df = pd.concat(frames, axis="columns")

            if favorability_col is not None and favorability:
                df["Favorability"] = favorability_col

            if flat_index:
                df = MetricsSummaryDisplay._flatten_index(df)

            return df

        estimator_name = df["estimator_name"].iloc[0]

        df = MetricsSummaryDisplay._frame_estimator(
            df, favorability=True, flat_index=False
        )
        favorability_col = df.pop("Favorability")

        if isinstance(aggregate, (list, tuple)):
            aggregate = list(aggregate)
        elif aggregate is not None:
            aggregate = cast(Literal["mean", "std"], aggregate)
            aggregate = [aggregate]

        df = df.reset_index().pivot_table(
            index=df.index.names,
            columns="split" if aggregate is None else None,
            values=estimator_name,
            aggfunc="first" if aggregate is None else aggregate,
            sort=False,
        )

        if aggregate is None:
            df.columns = pd.MultiIndex.from_product(
                [[estimator_name], [f"Split #{i}" for i in df.columns]]
            )
        else:
            df.columns = df.columns.swaplevel(0, 1)

        if favorability:
            df["Favorability"] = favorability_col[~favorability_col.index.duplicated()]

        if flat_index:
            df = MetricsSummaryDisplay._flatten_index(df)

        return df

    def frame(
        self,
        *,
        aggregate: Aggregate | None = ("mean", "std"),
        favorability: bool = False,
        flat_index: bool = False,
    ):
        """Return the summarize as a dataframe.

        Parameters
        ----------
        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Only used when `report_type` includes `"cross-validation"`.
            Functions to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.

        favorability : bool, default=False
            Whether or not to add an indicator of the favorability of the metric as
            an extra column in the returned DataFrame.

        flat_index : bool, default=False
            Whether to return a flat index or a multi-index.

        Returns
        -------
        frame : pandas.DataFrame
            The report metrics as a dataframe.
        """
        if self.report_type == "estimator":
            return MetricsSummaryDisplay._frame_estimator(
                self.data,
                favorability=favorability,
                flat_index=flat_index,
            )
        elif self.report_type == "cross-validation":
            return MetricsSummaryDisplay._frame_cross_validation(
                self.data,
                aggregate=aggregate,
                favorability=favorability,
                flat_index=flat_index,
            )

        elif self.report_type == "comparison-estimator":
            df = self.data.copy()

            df = pd.concat(
                [
                    MetricsSummaryDisplay._frame_estimator(
                        est, favorability=True, flat_index=False
                    )
                    for _, est in df.groupby("estimator_name", sort=False)
                ],
                axis="columns",
            )

            # Extract favorability columns and use first non-NaN value for each row
            favorability_col = df.pop("Favorability").bfill(axis=1).iloc[:, 0]

            df.columns.name = "Estimator"

            if favorability:
                df["Favorability"] = favorability_col

            if flat_index:
                df = MetricsSummaryDisplay._flatten_index(df)

            return df

        else:  # self.report_type == "comparison-cross-validation"
            df = self.data.copy()

            df = pd.concat(
                [
                    MetricsSummaryDisplay._frame_cross_validation(
                        est, aggregate=aggregate, favorability=True, flat_index=False
                    )
                    for _, est in df.groupby("estimator_name", sort=False)
                ],
                axis="columns",
            )

            # Sort columns to avoid lexsort warning when accessing specific columns
            df = df.sort_index(axis=1)
            favorability_col = df.pop(("Favorability", "")).bfill(axis=1).iloc[:, 0]

            if aggregate is None:
                df.columns.names = ["Estimator", "Split"]
            else:
                df.columns = df.columns.swaplevel(0, 1)
                df = df.sort_index(axis=1, level=[0, 1])
                df.columns.names = [None, "Estimator"]

            if favorability:
                df[("Favorability", "")] = favorability_col

            if flat_index:
                df = MetricsSummaryDisplay._flatten_index(df)

            return df

    def _repr_html_(self) -> str:
        return (
            f"{self.frame()._repr_html_()}"
            '<p role="note">Use <code>.frame()</code> to control the format'
            " of the output.</p>"
        )

    def __repr__(self) -> str:
        return f"{self.frame()!r}\nUse .frame() to control the format of the output."

    def _repr_mimebundle_(self, **kwargs):
        return {"text/plain": repr(self), "text/html": self._repr_html_()}

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        metric: str | list[str],
        subplot_by: Literal["auto", "estimator", "label", "output", "data_source"]
        | None = "auto",
    ) -> Figure:
        """Plot one or more metrics.

        Parameters
        ----------
        metric : str or list of str
            The metric(s) to plot, using the same registry keys as
            :meth:`~skore.EstimatorReport.metrics.summarize` (e.g. ``"precision"``).
            When several metrics are provided, they are shown on the same plot to
            compare trade-offs. Metrics with very different scales may look
            compressed on a shared x-axis.

        subplot_by : {"auto", "estimator", "label", "output", "data_source"} \
                or None, default="auto"
            The column to use for subplotting. If ``"auto"``, subplotting is
            performed only when comparing estimators in a multiclass classification
            or multi-output regression problem.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the metrics plot.
        """
        return self._plot(metric=metric, subplot_by=subplot_by)

    def _plot_matplotlib(
        self,
        *,
        metric: str | list[str],
        subplot_by: Literal["auto", "estimator", "label", "output", "data_source"]
        | None = "auto",
    ) -> Figure:
        """Dispatch the plotting function for matplotlib backend."""
        metrics = [metric] if isinstance(metric, str) else list(metric)
        frame = self._prepare_plot_frame(metrics)

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

        estimator_name = self.data["estimator_name"].iloc[0]
        return self._plot_single_estimator(
            frame=frame,
            estimator_name=estimator_name,
            report_type=self.report_type,
            subplot_by=subplot_by,
            barplot_kwargs=barplot_kwargs,
            boxplot_kwargs=boxplot_kwargs,
            stripplot_kwargs=stripplot_kwargs,
        )

    def _prepare_plot_frame(self, metrics: list[str]) -> pd.DataFrame:
        """Filter and reshape raw rows into a long frame for plotting."""
        if not metrics:
            raise ValueError(
                "At least one metric must be provided to plot. Pass a registry key "
                "such as `metric='precision'`."
            )

        frame = self.data.copy()
        available = set(frame["metric_name"])
        unknown = set(metrics) - available
        if unknown:
            raise ValueError(
                f"Unknown metric(s): {sorted(unknown)!r}. "
                f"Available metrics: {sorted(available)!r}."
            )

        frame = frame[frame["metric_name"].isin(metrics)]
        frame = frame.rename(columns={"estimator_name": "estimator"})

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

        verbose_order = (
            frame.drop_duplicates("metric_name")
            .set_index("metric_name")
            .loc[metrics, "metric_verbose_name"]
            .tolist()
        )
        frame["metric_verbose_name"] = pd.Categorical(
            frame["metric_verbose_name"],
            categories=verbose_order,
            ordered=True,
        )

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
        n_metrics: int,
        xlabel: str,
        ylabel: str = "",
    ) -> None:
        ax.set(xlabel=xlabel, ylabel=ylabel)
        for metric_idx in range(0, n_metrics, 2):
            ax.axhspan(
                metric_idx - 0.5,
                metric_idx + 0.5,
                color="lightgray",
                alpha=0.4,
                zorder=0,
            )

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
                y="metric_verbose_name",
                hue=hue,
                col=col,
                kind="bar",
                **(barplot_kwargs or {}),
            )
        else:
            facet = sns.catplot(
                data=frame,
                x="score",
                y="metric_verbose_name",
                hue=hue,
                col=col,
                kind="strip",
                dodge=True,
                **(stripplot_kwargs or {}),
            ).map_dataframe(
                sns.boxplot,
                x="score",
                y="metric_verbose_name",
                hue=hue,
                palette="tab10" if hue is not None else None,
                dodge=True,
                **(boxplot_kwargs or {}),
            )

        add_background_metrics = hue is not None
        figure = facet.figure
        ax_grid = facet.axes.squeeze()
        n_metrics = (
            [frame["metric_verbose_name"].nunique()]
            if col is None
            else [
                frame.query(f"{col} == @col_value")["metric_verbose_name"].nunique()
                for col_value in frame[col].unique()
            ]
        )
        xlabel = (
            frame["metric_verbose_name"].cat.categories[0]
            if frame["metric_verbose_name"].nunique() == 1
            else "Score"
        )
        for ax, n_metric in zip(ax_grid.flatten(), n_metrics, strict=True):
            self._decorate_matplotlib_axis(
                ax=ax,
                n_metrics=n_metric,
                xlabel=xlabel,
            )
            if not add_background_metrics:
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
