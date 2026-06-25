from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal, NotRequired, TypedDict, cast

import pandas as pd
from matplotlib.figure import Figure

from skore._sklearn._plot.base import DisplayMixin
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
    metric_name : str
        Technical metric name (e.g. ``"accuracy"``); matches the key under which
        the metric is registered in :attr:`EstimatorReport._metric_registry`.
    metric_verbose_name : str
        Human-readable metric name shown in the display.
    estimator_name : str
        Name shown in the display.
    data_source : {"train", "test"}
        Dataset split used to compute the metric.
    greater_is_better : bool or None
        Whether higher or lower values are better.
    fingerprint : str or None
        Identifier disambiguating distinct custom metrics that share
        ``metric_verbose_name``. ``None`` for built-in metrics.
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
    fingerprint: str | None
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
        metric scores and their metadata (e.g. ``metric_verbose_name``,
        ``estimator_name``, ``data_source``, ``label``, ``output``, ``average``,
        ``split``, ``score``).

    report_type : {"estimator", "comparison-estimator", "cross-validation", \
            "comparison-cross-validation"}
        The type of report.

    Attributes
    ----------
    summary : pandas.DataFrame
        The long-format dataframe storing the metric scores and metadata.
    report_type : ReportType
        The type of report.
    data : pandas.DataFrame
        The long-format summary with fingerprints resolved (read-only property).

    See Also
    --------
    EstimatorReport.metrics.summarize : Create this display from a report.
    RocCurveDisplay : Plot ROC curves.
    PrecisionRecallCurveDisplay : Plot precision-recall curves.
    ConfusionMatrixDisplay : Display the confusion matrix.
    PredictionErrorDisplay : Plot regression prediction error.
    """

    def __init__(
        self,
        summary: pd.DataFrame,
        report_type: ReportType,
    ):
        self.summary = summary
        self.report_type = report_type

    @classmethod
    def _from_rows(
        cls,
        rows: list[MetricsSummaryRow],
        report_type: ReportType,
    ) -> MetricsSummaryDisplay:
        """Build a display from metric rows, stored as a long-format DataFrame."""
        summary = pd.DataFrame(rows)

        if any(isinstance(r["label"], bool) for r in rows):
            summary["label"] = summary["label"].astype(pd.BooleanDtype())
        elif any(isinstance(r["label"], int) for r in rows):
            summary["label"] = summary["label"].astype(pd.Int64Dtype())

        if any(isinstance(r["output"], int) for r in rows):
            summary["output"] = summary["output"].astype(pd.Int64Dtype())

        return cls(summary, report_type=report_type)

    @property
    def rows(self) -> list[MetricsSummaryRow]:
        """Reconstruct metric rows from the stored summary dataframe."""
        nullable_cols = {
            "label",
            "average",
            "output",
            "greater_is_better",
            "fingerprint",
            "split",
        }
        rows: list[MetricsSummaryRow] = []
        for record in self.summary.to_dict("records"):
            row: dict[str, Any] = {}
            for key, value in record.items():
                if key in nullable_cols and pd.isna(value):
                    row[key] = None
                else:
                    row[key] = value
            rows.append(cast("MetricsSummaryRow", row))
        return rows

    @property
    def data(self):
        """Return the long-format summary with fingerprints resolved."""
        data = MetricsSummaryDisplay._resolve_fingerprints(self.summary)
        return data.drop(columns="fingerprint")

    @staticmethod
    def _resolve_fingerprints(data: pd.DataFrame) -> pd.DataFrame:
        """Disambiguate ``metric_verbose_name`` across distinct fingerprints.

        When several rows share a ``metric_verbose_name`` but come from metrics
        with different fingerprints, they are renamed ``{name}_1``, ``{name}_2``,
        ... in the order they first appear. ``None`` counts as a regular
        fingerprint value, so a custom metric reusing a built-in's display name
        will still get disambiguated against the built-in.

        Suffixes skip over any name already present in the column, so we don't
        produce a collision if a metric happens to already be called e.g.
        ``"Metric_1"``.
        """
        data = data.copy()

        # Fingerprint = str | np.nan
        # fingerprints_per_name: dict[MetricName, list[Fingerprint]]
        fingerprints_per_name = defaultdict(list)
        for name, fingerprint in (
            data[["metric_verbose_name", "fingerprint"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ):
            fingerprints_per_name[name].append(fingerprint)

        # Decide the new name for each (name, fingerprint) pair
        #
        # Suffixes skip over names already in the column

        # renaming: dict[(MetricName, Fingerprint), NewMetricName]
        renaming = {}
        metric_names = set(data["metric_verbose_name"])
        for name, fingerprints in fingerprints_per_name.items():
            if len(fingerprints) < 2:
                continue
            i = 0
            for fingerprint in fingerprints:
                while True:
                    i += 1
                    candidate = f"{name}_{i}"
                    if candidate not in metric_names:
                        break
                metric_names.add(candidate)
                renaming[(name, fingerprint)] = candidate

        for (name, fingerprint), new_name in renaming.items():
            fp_match = (
                data["fingerprint"].isna()
                if pd.isna(fingerprint)
                else data["fingerprint"] == fingerprint
            )
            data.loc[
                (data["metric_verbose_name"] == name) & fp_match, "metric_verbose_name"
            ] = new_name

        return data

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
        return MetricsSummaryDisplay(summary, report_type=report_type)

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

    def frame(self, *, favorability: bool = False) -> pd.DataFrame:
        """Return the summary as a tidy long-format dataframe.

        Parameters
        ----------
        favorability : bool, default=False
            Whether or not to add a ``favorability`` column indicating whether
            higher or lower values are better for each metric.

        Returns
        -------
        frame : pandas.DataFrame
            The report metrics in long format with a flat index. Columns are
            included depending on the report type and the available data:

            - ``estimator``: estimator name (comparison reports only)
            - ``split``: cross-validation split index (cross-validation reports only)
            - ``data_source``: data source (when more than one is present)
            - ``metric``: metric name
            - ``label``: class label (classification, when relevant)
            - ``output``: output index (multioutput regression, when relevant)
            - ``average``: averaging mode (when relevant)
            - ``value``: the metric value
            - ``favorability``: favorability indicator (when ``favorability=True``)
        """
        data = self.data

        columns: list[str] = []
        if "comparison" in self.report_type:
            columns.append("estimator_name")
        if "cross-validation" in self.report_type:
            columns.append("split")
        if data["data_source"].nunique() > 1:
            columns.append("data_source")
        columns.append("metric_verbose_name")
        columns.extend(
            col for col in ("label", "output", "average") if data[col].notna().any()
        )
        columns.append("score")

        frame = data[columns].copy()
        if favorability:
            frame["favorability"] = (
                data["greater_is_better"]
                .map({True: "(↗︎)", False: "(↘︎)"})
                .astype("string")
                .fillna("")
            )

        frame = frame.rename(
            columns={
                "estimator_name": "estimator",
                "metric_verbose_name": "metric",
                "score": "value",
            }
        )
        return frame.reset_index(drop=True)

    def _to_pivoted_frame(
        self,
        *,
        aggregate: Aggregate | None = ("mean", "std"),
        favorability: bool = False,
        flat_index: bool = False,
    ):
        """Return the summary as a pivoted, human-readable dataframe.

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
            The report metrics as a pivoted dataframe.
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
            f"{self._to_pivoted_frame()._repr_html_()}"
            '<p role="note">Use <code>.frame()</code> to control the format'
            " of the output.</p>"
        )

    def __repr__(self) -> str:
        return (
            f"{self._to_pivoted_frame()!r}"
            "\nUse .frame() to control the format of the output."
        )

    @DisplayMixin.style_plot
    def plot(self) -> Figure:
        """Plot the metrics summary (not implemented)."""
        raise NotImplementedError
