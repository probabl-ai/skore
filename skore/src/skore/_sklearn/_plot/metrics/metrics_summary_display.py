import itertools
from typing import Any, Literal, cast

import pandas as pd
from matplotlib.figure import Figure

from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn.types import Aggregate, ReportType
from skore._utils._index import flatten_multi_index
from skore._utils._metric_rows import rows_to_dataframe


class MetricsSummaryDisplay(DisplayMixin):
    """Display for summarize.

    An instance of this class will be created by `Report.metrics.summarize()`.
    This class should not be instantiated directly.

    Parameters
    ----------
    rows : list of dicts
        The rows to display.
        Expected keys:
        - "metric": human-readable metric name shown in the display.
        - "estimator_name"
        - "data_source": "train" or "test".
        - "score": numeric metric value (scalar).
        - "favorability": "(↗︎)", "(↘︎)" or "".

        Depending on the metric shape and report type, rows may also contain:
        - "label": class label for per-class classification metrics.
        - "output": output index for multioutput regression metrics.
        - "average": averaging mode when averaged over labels or outputs.
        - "split": cross-validation split index.

    report_type : {"estimator", "comparison-estimator", "cross-validation", \
            "comparison-cross-validation"}
        The type of report.
    """

    def __init__(
        self,
        rows: list[dict],
        report_type: ReportType,
    ):
        self.rows = rows
        self.report_type = report_type

    @property
    def data(self):
        return rows_to_dataframe(self.rows)

    @staticmethod
    def _concatenate(
        child_displays: list["MetricsSummaryDisplay"],
        *,
        report_type: ReportType,
        extra_rows_data: list[dict[str, Any]] | None = None,
    ):
        if extra_rows_data is None:
            extra_rows_data = [{} for _ in child_displays]

        rows = list(
            itertools.chain(
                *[
                    [row | extra_data for row in display.rows]
                    for display, extra_data in zip(
                        child_displays, extra_rows_data, strict=True
                    )
                ]
            )
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

        for col in df.columns.intersection(["label", "output", "average"]):
            df[col] = df[col].astype("str").replace("<NA>", "").fillna("")

        estimator_name = df.pop("estimator_name").iloc[0]
        index = df.columns.intersection(
            ["metric", "verbose_name", "label", "output", "average"]
        ).to_list()
        df = df.set_index(index)

        if not favorability:
            df = df.drop(columns="favorability")
        else:
            # Put favorability at the end
            df = df[
                [col for col in df.columns if col != "favorability"] + ["favorability"]
            ]

        # Rename columns as well as index names
        new_columns = {
            "metric": "Metric",
            "verbose_name": "Metric",
            "label": "Label / Average",
            "output": "Output",
            "average": "Average",
            "favorability": "Favorability",
            "score": estimator_name,
        }
        df = df.rename(columns=new_columns)
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
                df_pivoted["Favorability"] = df[df["data_source"] == "test"][
                    "Favorability"
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

            The returned columns depend on ``report_type`` and on the
            ``aggregate``/``favorability`` parameters described in the class notes.
            The row index always starts with ``"Metric"`` and may include additional
            levels for class labels, averaging modes, outputs, estimators, or splits.
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
