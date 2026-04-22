from typing import Any, Literal, NotRequired, TypedDict, cast

import pandas as pd
from matplotlib.figure import Figure

from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn.metrics import Metric
from skore._sklearn.types import (
    Aggregate,
    DataSource,
    MLTask,
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

    metric_verbose_name: str
    estimator_name: str
    data_source: DataSource
    greater_is_better: bool | None
    score: Any
    label: PositiveLabel | None
    average: str | None
    output: int | None
    split: NotRequired[int]


def metric_score_to_rows(
    score: float | list | dict,
    *,
    metric: Metric,
    ml_task: MLTask,
    data_source: DataSource,
    estimator_name: str,
    pos_label: PositiveLabel = None,
    kwargs: dict[str, Any] | None = None,
) -> list[MetricsSummaryRow]:
    """Expand a metric score into display rows based on the ML task.

    Parameters
    ----------
    score : float, dict, or list
        The metric score.

    metric : Metric
        The metric instance (provides ``verbose_name``, ``icon``,
        and default ``kwargs``).

    ml_task : str
        The ML task (e.g. ``"binary-classification"``).

    data_source : {"test", "train"}
        The data source to use.

    estimator_name : str
        Name shown in the display.

    pos_label : label, default=None
        Positive label for binary classification.

    kwargs : dict, optional
        Keyword arguments used for the score call. Default is ``metric.kwargs``.
    """
    if kwargs is None:
        kwargs = metric.kwargs

    row: MetricsSummaryRow = {
        "metric_verbose_name": metric.verbose_name,
        "estimator_name": estimator_name,
        "data_source": data_source,
        "greater_is_better": metric.greater_is_better,
        "label": None,
        "average": None,
        "output": None,
        "score": score,
    }

    if ml_task == "binary-classification" and kwargs.get("average") == "binary":
        return [{**row, "label": kwargs.get("pos_label", pos_label)}]
    if ml_task in ("binary-classification", "multiclass-classification"):
        if isinstance(score, dict):
            return [{**row, "label": label, "score": score[label]} for label in score]
        return [{**row, "average": kwargs.get("average")}]
    if ml_task == "multioutput-regression":
        if isinstance(score, list):
            return [{**row, "output": idx, "score": s} for idx, s in enumerate(score)]
        return [{**row, "average": kwargs.get("multioutput")}]
    return [row]


class MetricsSummaryDisplay(DisplayMixin):
    """Display for summarize.

    An instance of this class will be created by `Report.metrics.summarize()`.
    This class should not be instantiated directly.

    Parameters
    ----------
    rows : list of MetricsSummaryRow
        The rows to display.

    report_type : {"estimator", "comparison-estimator", "cross-validation", \
            "comparison-cross-validation"}
        The type of report.
    """

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
        child_displays: list["MetricsSummaryDisplay"],
        *,
        report_type: ReportType,
        extra_rows_data: list[dict[str, Any]],
    ) -> "MetricsSummaryDisplay":
        rows = []
        for display, extra_data in zip(child_displays, extra_rows_data, strict=True):
            rows.extend(
                [cast(MetricsSummaryRow, row | extra_data) for row in display.rows]
            )

        return MetricsSummaryDisplay(rows, report_type=report_type)

    @staticmethod
    def _combine_display_column(
        df: pd.DataFrame,
        *,
        primary: str,
        secondary: str,
        combined: str,
    ) -> pd.DataFrame:
        overlap = df[primary].notna() & df[secondary].notna()
        if overlap.any():
            raise ValueError(
                f"Expected '{primary}' and '{secondary}' to be mutually exclusive."
            )

        df = df.copy()
        df[combined] = df[primary].combine_first(df[secondary])
        return df.drop(columns=[primary, secondary])

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

        extra_index_name = None
        if {"label", "average"}.issubset(df.columns):
            df = MetricsSummaryDisplay._combine_display_column(
                df,
                primary="label",
                secondary="average",
                combined="label_or_average",
            )
            extra_index_name = "label_or_average"
        elif {"output", "average"}.issubset(df.columns):
            df = MetricsSummaryDisplay._combine_display_column(
                df,
                primary="output",
                secondary="average",
                combined="output_or_average",
            )
            extra_index_name = "output_or_average"
        elif "label" in df.columns:
            extra_index_name = "label"
        elif "output" in df.columns:
            extra_index_name = "output"
        elif "average" in df.columns:
            extra_index_name = "average"

        for col in df.columns.intersection(
            ["label", "output", "average", "label_or_average", "output_or_average"]
        ):
            df[col] = df[col].astype("string").fillna("")

        estimator_name = df.pop("estimator_name").iloc[0]

        index = ["metric_verbose_name"]
        if extra_index_name is not None:
            index.append(extra_index_name)
        df = df.set_index(index)

        # Rename columns as well as index names
        new_columns = {
            "metric_verbose_name": "Metric",
            "label": "Label",
            "output": "Output",
            "label_or_average": "Label / Average",
            "output_or_average": "Output / Average",
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
                original_index_names = list(df.index.names)
                df = df.stack([0, 1])
                df.index.names = original_index_names + ["Estimator", "Split"]
                df = df.to_frame("Value")
                df.columns.name = None
            else:
                df.columns = df.columns.swaplevel(0, 1)
                df = df.sort_index(axis=1, level=[0, 1])
                df.columns.names = [None, "Estimator"]

            if favorability:
                df[("Favorability", "")] = favorability_col

            if flat_index:
                df = MetricsSummaryDisplay._flatten_index(df)

            return df

    @DisplayMixin.style_plot
    def plot(self) -> Figure:
        """Not yet implemented."""
        raise NotImplementedError
