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

FrameFormat = Literal["long", "wide", "auto"]


class MetricsSummaryRow(TypedDict):
    """A single row rendered by ``MetricsSummaryDisplay``.

    Parameters
    ----------
    name : str
        Technical metric name (e.g. ``"accuracy"``); matches the key under which
        the metric is registered in :attr:`EstimatorReport._metric_registry`.
    verbose_name : str
        Human-readable metric name shown in the display.
    estimator_name : str
        Name shown in the display.
    data_source : {"train", "test"}
        Dataset split used to compute the metric.
    greater_is_better : bool or None
        Whether higher or lower values are better.
    fingerprint : str or None
        Identifier disambiguating distinct custom metrics that share
        ``verbose_name``. ``None`` for built-in metrics.
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
        metric scores and their metadata (e.g. ``name``, ``verbose_name``,
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
    def _compute_data_for_display(
        cls,
        rows: list[MetricsSummaryRow],
        *,
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

    @staticmethod
    def _finalize_summary(summary: pd.DataFrame) -> pd.DataFrame:
        """Resolve fingerprint collisions and drop the fingerprint column."""
        if "fingerprint" not in summary.columns:
            return summary
        summary = MetricsSummaryDisplay._resolve_fingerprints(summary)
        return summary.drop(columns="fingerprint")

    @classmethod
    def _finalize(cls, display: MetricsSummaryDisplay) -> MetricsSummaryDisplay:
        """Return a display with a display-ready summary."""
        return cls(
            summary=cls._finalize_summary(display.summary),
            report_type=display.report_type,
        )

    @staticmethod
    def _resolve_fingerprints(data: pd.DataFrame) -> pd.DataFrame:
        """Disambiguate ``verbose_name`` across distinct fingerprints.

        When several rows share a ``verbose_name`` but come from metrics
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
            data[["verbose_name", "fingerprint"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ):
            fingerprints_per_name[name].append(fingerprint)

        # Decide the new name for each (name, fingerprint) pair
        #
        # Suffixes skip over names already in the column

        # renaming: dict[(MetricName, Fingerprint), NewMetricName]
        renaming = {}
        verbose_names = set(data["verbose_name"])
        for name, fingerprints in fingerprints_per_name.items():
            if len(fingerprints) < 2:
                continue
            i = 0
            for fingerprint in fingerprints:
                while True:
                    i += 1
                    candidate = f"{name}_{i}"
                    if candidate not in verbose_names:
                        break
                verbose_names.add(candidate)
                renaming[(name, fingerprint)] = candidate

        for (name, fingerprint), new_name in renaming.items():
            fp_match = (
                data["fingerprint"].isna()
                if pd.isna(fingerprint)
                else data["fingerprint"] == fingerprint
            )
            data.loc[(data["verbose_name"] == name) & fp_match, "verbose_name"] = (
                new_name
            )

        return data

    @staticmethod
    def _concatenate(
        child_displays: list[MetricsSummaryDisplay],
        *,
        report_type: ReportType,
        extra_rows_data: list[dict[str, Any]],
        finalize: bool = True,
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
        if finalize:
            summary = MetricsSummaryDisplay._finalize_summary(summary)
        return MetricsSummaryDisplay(summary, report_type=report_type)

    @classmethod
    def _resolve_format(
        cls,
        format: FrameFormat,
        report_type: ReportType,
    ) -> Literal["long", "wide"]:
        """Resolve ``format`` to either ``long`` or ``wide``."""
        if format == "long":
            return "long"

        if format == "wide":
            return "wide"

        if format == "auto":
            return "long" if "comparison" in report_type else "wide"

        raise ValueError(
            f"Invalid format: {format!r}. Expected 'long', 'wide', or 'auto'."
        )

    @staticmethod
    def _normalize_aggregate(
        aggregate: Aggregate | None,
    ) -> list[Literal["mean", "std"]] | None:
        if aggregate is None:
            return None
        if isinstance(aggregate, (list, tuple)):
            return list(aggregate)
        return [cast(Literal["mean", "std"], aggregate)]

    @staticmethod
    def _metric_column(*, verbose_name: bool) -> Literal["verbose_name", "name"]:
        return "verbose_name" if verbose_name else "name"

    @staticmethod
    def _favorability_column(series: pd.Series) -> pd.Series:
        return series.map({True: "(↗︎)", False: "(↘︎)"}).fillna("").astype("string")

    @staticmethod
    def _finalize_wide(
        df: pd.DataFrame, *, with_multiindex: bool, verbose_name: bool = False
    ) -> pd.DataFrame:
        if with_multiindex:
            return df
        return MetricsSummaryDisplay._flatten_index(df, lowercase=not verbose_name)

    @staticmethod
    def _flatten_index(df: pd.DataFrame, *, lowercase: bool = True) -> pd.DataFrame:
        df = df.copy()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = flatten_multi_index(df.columns, lowercase=lowercase)
        if isinstance(df.index, pd.MultiIndex):
            df.index = flatten_multi_index(df.index, lowercase=lowercase)
        if isinstance(df.index, pd.Index):
            df.index = df.index.str.replace(r"\((.*)\)$", r"\1", regex=True)

        return df

    @staticmethod
    def _frame_estimator(
        data: pd.DataFrame,
        *,
        favorability: bool = False,
        verbose_name: bool = False,
    ) -> pd.DataFrame:
        """Process estimator report data into a formatted dataframe."""
        df = data.copy()
        df = df.dropna(axis="columns", how="all")
        metric_col = MetricsSummaryDisplay._metric_column(verbose_name=verbose_name)
        other_col = "name" if verbose_name else "verbose_name"
        df = df.drop(columns=other_col, errors="ignore")

        for col in df.columns.intersection(["label", "output", "average"]):
            df[col] = df[col].astype("string").fillna("")

        estimator_name = df.pop("estimator_name").iloc[0]
        index = df.columns.intersection(
            [metric_col, "label", "output", "average"]
        ).to_list()
        df = df.set_index(index)

        # Rename columns as well as index names
        new_columns = {
            metric_col: "Metric",
            "label": "Label",
            "output": "Output",
            "average": "Average",
            "score": estimator_name,
        }
        df = df.rename(columns=new_columns)

        if favorability:
            df["Favorability"] = MetricsSummaryDisplay._favorability_column(
                df["greater_is_better"]
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

        return df

    @staticmethod
    def _frame_cross_validation(
        data: pd.DataFrame,
        *,
        aggregate: Aggregate | None = ("mean", "std"),
        favorability: bool = False,
        verbose_name: bool = False,
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
                    verbose_name=verbose_name,
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

            return df

        estimator_name = df["estimator_name"].iloc[0]

        df = MetricsSummaryDisplay._frame_estimator(
            df, favorability=True, verbose_name=verbose_name
        )
        favorability_col = df.pop("Favorability")

        aggregate = MetricsSummaryDisplay._normalize_aggregate(aggregate)

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

        return df

    @staticmethod
    def _frame_comparison(
        summary: pd.DataFrame,
        *,
        report_type: ReportType,
        aggregate: Aggregate | None,
        favorability: bool,
        verbose_name: bool,
    ) -> pd.DataFrame:
        """Process comparison report data into a formatted dataframe."""
        if report_type == "comparison-estimator":
            df = pd.concat(
                [
                    MetricsSummaryDisplay._frame_estimator(
                        est, favorability=True, verbose_name=verbose_name
                    )
                    for _, est in summary.groupby("estimator_name", sort=False)
                ],
                axis="columns",
            )
            favorability_col = df.pop("Favorability").bfill(axis=1).iloc[:, 0]
            df.columns.name = "Estimator"
            if favorability:
                df["Favorability"] = favorability_col
            return df

        df = pd.concat(
            [
                MetricsSummaryDisplay._frame_cross_validation(
                    est,
                    aggregate=aggregate,
                    favorability=True,
                    verbose_name=verbose_name,
                )
                for _, est in summary.groupby("estimator_name", sort=False)
            ],
            axis="columns",
        )
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
        return df

    def _to_long_frame(
        self,
        *,
        favorability: bool = False,
        verbose_name: bool = False,
        aggregate: Aggregate | None = ("mean", "std"),
    ) -> pd.DataFrame:
        """Return the summary as a tidy long-format dataframe.

        Parameters
        ----------
        favorability : bool, default=False
            Whether or not to add a ``favorability`` column indicating whether
            higher or lower values are better for each metric.

        verbose_name : bool, default=False
            If ``True``, the ``metric`` column contains human-readable names.
            If ``False``, the ``metric`` column contains technical registry names.

        aggregate : {"mean", "std"}, list of such str or None, \
                default=("mean", "std")
            For cross-validation reports, controls whether rows are aggregated
            across splits (with an ``aggregate`` column) or listed per split
            (with a ``split`` column when ``aggregate`` is ``None``).

        Returns
        -------
        frame : pandas.DataFrame
            The report metrics in long format with a flat index. Columns are
            included depending on the report type and the available data:

            - ``estimator``: estimator name (comparison reports only)
            - ``split``: cross-validation split index (CV, ``aggregate=None``)
            - ``aggregate``: aggregation statistic (CV, when aggregating)
            - ``data_source``: data source (when more than one is present)
            - ``metric``: metric name (verbose or technical, per ``verbose_name``)
            - ``label``: class label (classification, when relevant)
            - ``output``: output index (multioutput regression, when relevant)
            - ``average``: averaging mode (when relevant)
            - ``value``: the metric value
            - ``favorability``: favorability indicator (when ``favorability=True``)
        """
        if (
            "cross-validation" in self.report_type
            and self._normalize_aggregate(aggregate) is not None
        ):
            return self._to_long_frame_cv_aggregate(
                favorability=favorability,
                verbose_name=verbose_name,
                aggregate=aggregate,
            )

        data = self.summary
        metric_col = self._metric_column(verbose_name=verbose_name)

        columns: list[str] = []
        if "comparison" in self.report_type:
            columns.append("estimator_name")
        if "cross-validation" in self.report_type:
            columns.append("split")
        if data["data_source"].nunique() > 1:
            columns.append("data_source")
        columns.append(metric_col)
        columns.extend(
            col for col in ("label", "output", "average") if data[col].notna().any()
        )
        columns.append("score")

        frame = data[columns].copy()
        if favorability:
            frame["favorability"] = self._favorability_column(data["greater_is_better"])

        frame = frame.rename(
            columns={
                "estimator_name": "estimator",
                metric_col: "metric",
                "score": "value",
            }
        )
        return frame.reset_index(drop=True)

    def _to_long_frame_cv_aggregate(
        self,
        *,
        favorability: bool,
        verbose_name: bool,
        aggregate: Aggregate | None,
    ) -> pd.DataFrame:
        """Return aggregated long-format rows for cross-validation reports."""
        data = self.summary
        metric_col = self._metric_column(verbose_name=verbose_name)
        agg_funcs = self._normalize_aggregate(aggregate)
        assert agg_funcs is not None

        group_cols: list[str] = []
        if "comparison" in self.report_type:
            group_cols.append("estimator_name")
        if data["data_source"].nunique() > 1:
            group_cols.append("data_source")
        group_cols.append(metric_col)
        group_cols.extend(
            col for col in ("label", "output", "average") if data[col].notna().any()
        )

        favorability_by_group = None
        if favorability:
            favorability_by_group = data.groupby(group_cols, dropna=False)[
                "greater_is_better"
            ].first()

        frames = []
        for agg_name in agg_funcs:
            aggregated = (
                data.groupby(group_cols, dropna=False)["score"]
                .agg(agg_name)
                .reset_index(name="value")
            )
            aggregated["aggregate"] = agg_name
            frames.append(aggregated)

        frame = pd.concat(frames, ignore_index=True)
        if favorability and favorability_by_group is not None:
            frame = frame.merge(
                favorability_by_group.rename("greater_is_better"),
                on=group_cols,
                how="left",
            )
            frame["favorability"] = self._favorability_column(
                frame["greater_is_better"]
            )
            frame = frame.drop(columns="greater_is_better")

        return frame.rename(
            columns={
                "estimator_name": "estimator",
                metric_col: "metric",
            }
        ).reset_index(drop=True)

    def _to_wide_frame(
        self,
        *,
        aggregate: Aggregate | None = ("mean", "std"),
        favorability: bool = False,
        verbose_name: bool = False,
        with_multiindex: bool = False,
    ) -> pd.DataFrame:
        """Return the summary as a pivoted, flattened dataframe."""
        if self.report_type == "estimator":
            df = MetricsSummaryDisplay._frame_estimator(
                self.summary,
                favorability=favorability,
                verbose_name=verbose_name,
            )
        elif self.report_type == "cross-validation":
            df = MetricsSummaryDisplay._frame_cross_validation(
                self.summary,
                aggregate=aggregate,
                favorability=favorability,
                verbose_name=verbose_name,
            )
        else:
            df = MetricsSummaryDisplay._frame_comparison(
                self.summary,
                report_type=self.report_type,
                aggregate=aggregate,
                favorability=favorability,
                verbose_name=verbose_name,
            )

        return self._finalize_wide(
            df, with_multiindex=with_multiindex, verbose_name=verbose_name
        )

    def _repr_frame(self, *, for_html: bool = False) -> pd.DataFrame:
        """Return the dataframe used for display representation."""
        aggregate = cast(Aggregate, ("mean", "std"))
        is_comparison = "comparison" in self.report_type
        return self.frame(
            format="auto",
            aggregate=aggregate,
            with_multiindex=for_html and not is_comparison,
        )

    def frame(
        self,
        *,
        format: FrameFormat = "auto",
        favorability: bool = False,
        verbose_name: bool = False,
        with_multiindex: bool = False,
        aggregate: Aggregate | None = ("mean", "std"),
    ) -> pd.DataFrame:
        """Return the metrics summary as a dataframe.

        Parameters
        ----------
        format : {"long", "wide", "auto"}, default="auto"
            Output shape:

            - ``"long"``: one row per observation with a flat index.
            - ``"wide"``: pivoted, human-readable layout with flat index and
              columns (no MultiIndex) by default.
            - ``"auto"``: long for comparison reports, wide for estimator and
              cross-validation reports.

        favorability : bool, default=False
            Whether to add a favorability indicator column.

        verbose_name : bool, default=False
            Whether metric identifiers use human-readable names (``verbose_name``)
            or technical registry names (``name``). Applies to both long and
            wide formats.

        with_multiindex : bool, default=False
            Only used when the resolved format is ``"wide"``. If ``True``,
            preserve row and column MultiIndex levels instead of flattening
            them to single-level strings.

        aggregate : {"mean", "std"}, list of such str or None, \
                default=("mean", "std")
            For cross-validation reports, controls split aggregation in both
            long and wide formats. Ignored for estimator reports.

        Returns
        -------
        pandas.DataFrame
            The formatted metrics summary.
        """
        resolved = self._resolve_format(format, self.report_type)
        if resolved == "long":
            return self._to_long_frame(
                favorability=favorability,
                verbose_name=verbose_name,
                aggregate=aggregate,
            )
        return self._to_wide_frame(
            favorability=favorability,
            aggregate=aggregate,
            verbose_name=verbose_name,
            with_multiindex=with_multiindex,
        )

    def _repr_html_(self) -> str:
        return (
            f"{self._repr_frame(for_html=True)._repr_html_()}"
            '<p role="note">Use <code>.frame()</code> to control the format'
            " of the output.</p>"
        )

    def __repr__(self) -> str:
        return (
            f"{self._repr_frame()!r}\nUse .frame() to control the format of the output."
        )

    @DisplayMixin.style_plot
    def plot(self) -> Figure:
        """Plot the metrics summary (not implemented)."""
        raise NotImplementedError
