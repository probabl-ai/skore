from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal, NotRequired, TypedDict, cast

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.utils.validation import _is_arraylike

from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn.types import (
    Aggregate,
    DataSource,
    PositiveLabel,
    ReportType,
)
from skore._utils._index import flatten_multi_index, squeeze_single_column

FrameFormat = Literal["long", "wide", "auto"]

_FAVORABILITY_SYMBOLS = {True: "(↗︎)", False: "(↘︎)"}

_INDEX_LEVEL_NAMES = {
    "metric": "Metric",
    "label": "Label",
    "output": "Output",
    "average": "Average",
}


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

        if "fingerprint" not in summary.columns:
            summary["fingerprint"] = None

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

        return cls(summary, report_type=report_type)

    @staticmethod
    def _resolve_fingerprints(data: pd.DataFrame) -> pd.DataFrame:
        """Disambiguate metric names across distinct fingerprints.

        When several rows share a name (``name`` or ``verbose_name``) but have
        different fingerprints, they are renamed ``{name}_1``, ``{name}_2``, ...
        in the order they first appear. Built-in metrics use a ``None``
        fingerprint; a custom metric that reuses a built-in name is still
        disambiguated because its fingerprint differs from ``None``.

        Both ``name`` and ``verbose_name`` are disambiguated independently, so
        whichever column :meth:`frame` displays is unambiguous.

        Suffixes skip over any name already present in the column, so we don't
        produce a collision if a metric happens to already be called e.g.
        ``"Metric_1"``.
        """
        data = data.copy()

        for name_col in ("name", "verbose_name"):
            fingerprints_per_name = defaultdict(list)
            for name, fingerprint in (
                data[[name_col, "fingerprint"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            ):
                fingerprints_per_name[name].append(fingerprint)

            renaming = {}
            existing = set(data[name_col])
            for name, fingerprints in fingerprints_per_name.items():
                if len(fingerprints) < 2:
                    continue
                i = 0
                for fingerprint in fingerprints:
                    while True:
                        i += 1
                        candidate = f"{name}_{i}"
                        if candidate not in existing:
                            break
                    existing.add(candidate)
                    renaming[(name, fingerprint)] = candidate

            for (name, fingerprint), new_name in renaming.items():
                fp_match = (
                    data["fingerprint"].isna()
                    if pd.isna(fingerprint)
                    else data["fingerprint"] == fingerprint
                )
                data.loc[(data[name_col] == name) & fp_match, name_col] = new_name

        return data

    def _prepare_pivot_df(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Return a pivot-ready frame and the columns used as the row index.

        Returns
        -------
        pivot_df : pandas.DataFrame
            Copy of ``df`` ready for pivoting.
        index_cols : list of str
            Columns that form the row index, always starting with ``metric``.
        """
        index_cols = ["metric"] + [
            col
            for col in ("label", "output", "average")
            if col in df.columns and df[col].notna().any()
        ]
        pivot_df = df.copy()
        for col in ("label", "output", "average"):
            if col in pivot_df.columns:
                # pandas pivots drop or reorder ``pd.NA`` index keys; use a
                # sentinel so every dimension level is preserved.
                pivot_df[col] = pivot_df[col].astype(object).fillna("")
        return pivot_df, index_cols

    @staticmethod
    def _to_favorability(greater_is_better: pd.Series) -> pd.Series:
        """Map ``greater_is_better`` flags to favorability arrow symbols."""
        return greater_is_better.map(_FAVORABILITY_SYMBOLS).fillna("").astype("string")

    def _prepare_long(
        self,
        summary: pd.DataFrame,
        *,
        verbose_name: bool = False,
        aggregate: Aggregate | None = ("mean", "std"),
    ) -> pd.DataFrame:
        """Normalize ``summary`` into a long table for export or pivoting."""
        metric_col = "verbose_name" if verbose_name else "name"

        if "cross-validation" in self.report_type and aggregate is not None:
            group_cols: list[str] = []
            if "comparison" in self.report_type:
                group_cols.append("estimator_name")
            if summary["data_source"].nunique() > 1:
                group_cols.append("data_source")
            group_cols.append(metric_col)
            group_cols.extend(
                col
                for col in ("label", "output", "average")
                if summary[col].notna().any()
            )

            aggregated = summary.groupby(group_cols, dropna=False, sort=False)[
                "score"
            ].agg(aggregate)
            if isinstance(aggregated, pd.Series):
                frame = aggregated.rename("value").reset_index()
                frame["aggregate"] = aggregate
            else:
                frame = (
                    aggregated.stack(future_stack=True).rename("value").reset_index()
                )
                frame = frame.rename(columns={frame.columns[-2]: "aggregate"})

            frame["greater_is_better"] = frame.merge(
                summary.groupby(group_cols, dropna=False, sort=False)[
                    "greater_is_better"
                ].first(),
                on=group_cols,
                how="left",
            )["greater_is_better"]

            return frame.rename(
                columns={"estimator_name": "estimator", metric_col: "metric"}
            ).reset_index(drop=True)

        columns: list[str] = []
        if "comparison" in self.report_type:
            columns.append("estimator_name")
        if "cross-validation" in self.report_type:
            columns.append("split")
        if summary["data_source"].nunique() > 1:
            columns.append("data_source")
        columns.append(metric_col)
        columns.extend(
            col for col in ("label", "output", "average") if summary[col].notna().any()
        )
        columns.extend(["score", "greater_is_better"])

        frame = summary[columns].copy()
        return frame.rename(
            columns={
                "estimator_name": "estimator",
                metric_col: "metric",
                "score": "value",
            }
        ).reset_index(drop=True)

    def _long_to_wide(
        self,
        df: pd.DataFrame,
        *,
        aggregate: Aggregate | None = ("mean", "std"),
        favorability: bool = False,
        verbose_name: bool = False,
        flat_index: bool = True,
    ) -> pd.DataFrame:
        """Pivot a prepared long table into wide layout."""
        if self.report_type == "estimator":
            wide = self._long_to_wide_estimator(df)
        elif self.report_type == "cross-validation":
            wide = self._long_to_wide_cross_validation(df, aggregate=aggregate)
        elif self.report_type == "comparison-estimator":
            wide = self._long_to_wide_comparison_estimator(df)
        else:
            wide = self._long_to_wide_comparison_cross_validation(
                df, aggregate=aggregate
            )

        if favorability:
            # ``greater_is_better`` is constant per metric row (it does not vary
            # with ``data_source``, ``split``, ``aggregate`` or ``estimator``), so
            # a single ``groupby(index_cols).first()`` yields the value for every
            # wide layout.
            pivot_df, index_cols = self._prepare_pivot_df(df)
            favorability_col = self._to_favorability(
                pivot_df.groupby(index_cols, dropna=False)["greater_is_better"].first()
            )
            favorability_col.index = favorability_col.index.set_names(wide.index.names)
            wide["Favorability"] = favorability_col

        if flat_index:
            lowercase = not verbose_name
            if isinstance(wide.columns, pd.MultiIndex):
                wide.columns = flatten_multi_index(wide.columns, lowercase=lowercase)
            if isinstance(wide.index, pd.MultiIndex):
                wide.index = flatten_multi_index(wide.index, lowercase=lowercase)
            if isinstance(wide.index, pd.Index):
                wide.index = wide.index.str.replace(r"\((.*)\)$", r"\1", regex=True)
        elif isinstance(wide.index, pd.MultiIndex):
            levels = list(wide.index.levels)
            for level_index, name in enumerate(wide.index.names):
                if name == "Label":
                    levels[level_index] = pd.Index(
                        [
                            "" if value == "" else str(value)
                            for value in levels[level_index]
                        ],
                        dtype="string",
                        name=name,
                    )
            wide.index = wide.index.set_levels(levels)

        return wide

    def _long_to_wide_estimator(
        self,
        df: pd.DataFrame,
        *,
        estimator_name: str | None = None,
    ) -> pd.DataFrame:
        """Pivot an estimator report into wide layout (one value column)."""
        pivot_df, index_cols = self._prepare_pivot_df(df)
        if estimator_name is None:
            estimator_name = self.summary["estimator_name"].iloc[0]

        if "data_source" not in df.columns or df["data_source"].nunique() == 1:
            wide = pivot_df.set_index(index_cols)[["value"]]
            wide.columns = [estimator_name]
        else:
            wide = pivot_df.pivot_table(
                index=index_cols,
                columns="data_source",
                values="value",
                aggfunc="first",
                sort=False,
            )
            wide = wide[["train", "test"]]
            wide.columns = [f"{estimator_name} ({col})" for col in wide.columns]

        wide.index = wide.index.set_names(
            [_INDEX_LEVEL_NAMES[col] for col in index_cols]
        )
        return wide

    def _long_to_wide_cross_validation(
        self,
        df: pd.DataFrame,
        *,
        aggregate: Aggregate | None,
        estimator_name: str | None = None,
    ) -> pd.DataFrame:
        """Pivot a cross-validation report into wide layout (split/aggregate cols)."""
        if "data_source" in df.columns and df["data_source"].nunique() > 1:
            frames = []
            for data_source in ("train", "test"):
                source_df = df[df["data_source"] == data_source]
                source_frame = self._long_to_wide_cross_validation(
                    source_df,
                    aggregate=aggregate,
                    estimator_name=estimator_name,
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

            return pd.concat(frames, axis="columns")

        pivot_df, index_cols = self._prepare_pivot_df(df)
        if estimator_name is None:
            estimator_name = self.summary["estimator_name"].iloc[0]

        if "aggregate" in df.columns:
            wide = pivot_df.pivot_table(
                index=index_cols,
                columns="aggregate",
                values="value",
                aggfunc="first",
                sort=False,
            )
            if isinstance(wide.columns, pd.MultiIndex):
                wide.columns = wide.columns.swaplevel(0, 1)
            elif isinstance(aggregate, str):
                wide.columns = pd.MultiIndex.from_tuples([(estimator_name, aggregate)])
            else:
                wide.columns = pd.MultiIndex.from_tuples(
                    [(estimator_name, str(col)) for col in wide.columns]
                )
        else:
            wide = pivot_df.pivot_table(
                index=index_cols,
                columns="split",
                values="value",
                aggfunc="first",
                sort=False,
            )
            wide.columns = pd.MultiIndex.from_product(
                [[estimator_name], [f"Split #{i}" for i in wide.columns]]
            )

        wide.index = wide.index.set_names(
            [_INDEX_LEVEL_NAMES[col] for col in index_cols]
        )
        return wide

    def _long_to_wide_comparison_estimator(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Pivot a comparison-estimator report into wide layout."""
        wide = pd.concat(
            [
                self._long_to_wide_estimator(
                    estimator_df.drop(columns="estimator"),
                    estimator_name=estimator_key,
                )
                for estimator_key, estimator_df in df.groupby("estimator", sort=False)
            ],
            axis="columns",
        )
        wide.columns.name = "Estimator"
        return wide

    def _long_to_wide_comparison_cross_validation(
        self,
        df: pd.DataFrame,
        *,
        aggregate: Aggregate | None,
    ) -> pd.DataFrame:
        """Pivot a comparison-cross-validation report into wide layout."""
        estimators: list[str] = []
        frames = []
        for estimator_key, estimator_df in df.groupby("estimator", sort=False):
            estimators.append(estimator_key)
            frames.append(
                self._long_to_wide_cross_validation(
                    estimator_df.drop(columns="estimator"),
                    aggregate=aggregate,
                    estimator_name=estimator_key,
                )
            )

        wide = pd.concat(frames, axis="columns")

        if aggregate is None:
            wide.columns.names = ["Estimator", "Split"]
        else:
            # Group the aggregate level (e.g. mean, std) together while keeping
            # estimators in the order they were passed to the comparison.
            wide = wide.swaplevel(0, 1, axis="columns")
            aggregate_order = list(dict.fromkeys(wide.columns.get_level_values(0)))
            wide = wide[
                [
                    (agg, estimator)
                    for agg in aggregate_order
                    for estimator in estimators
                ]
            ]
            wide.columns.names = [None, "Estimator"]

        return wide

    def frame(
        self,
        *,
        format: FrameFormat = "auto",
        favorability: bool = False,
        verbose_name: bool = False,
        flat_index: bool = True,
        aggregate: Aggregate | None = ("mean", "std"),
    ) -> pd.DataFrame | pd.Series:
        """Return the metrics summary as a table.

        Parameters
        ----------
        format : {"auto", "long", "wide"}, default="auto"
            The shape of the returned object. ``"auto"`` resolves to
            ``"wide"`` for estimator and cross-validation reports, and to
            ``"long"`` for comparison reports. ``"long"`` returns one row per
            metric observation, while ``"wide"`` pivots the metrics into a
            tabular layout.

        favorability : bool, default=False
            Whether to add a column indicating whether higher ``(↗︎)`` or lower
            ``(↘︎)`` values are better for each metric.

        verbose_name : bool, default=False
            Whether to use the human-readable metric names instead of the
            technical names (e.g. ``"Accuracy"`` instead of ``"accuracy"``).

        flat_index : bool, default=True
            Whether to keep the row and column MultiIndex.
            When ``False``, the index and columns are flattened into a
            single level. Has no effect when ``format="long"``.

        aggregate : {"mean", "std"}, list of such str or None, \
                default=("mean", "std")
            Only used for cross-validation reports. Functions to aggregate the
            scores across the cross-validation splits. ``None`` returns the
            scores for each split.

        Returns
        -------
        pandas.DataFrame or pandas.Series
            The metrics summary. The shape depends on ``format``: ``"long"``
            yields one row per metric observation, whereas ``"wide"`` pivots the
            metrics into a table whose columns depend on the report type. For
            ``"wide"`` layouts with a single value column, a :class:`pandas.Series`
            is returned with its name set to that column label.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> estimator = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(estimator, X, y)
        >>> metrics = report.metrics.summarize().frame()
        >>> metrics.loc["accuracy"]  # Series for single-estimator wide layout
        """
        if format not in {"long", "wide", "auto"}:
            raise ValueError(
                f"Invalid format: {format!r}. Expected 'long', 'wide', or 'auto'."
            )
        if format == "auto":
            resolved = "long" if "comparison" in self.report_type else "wide"
        else:
            resolved = format

        summary = self._resolve_fingerprints(self.summary)
        prepared = self._prepare_long(
            summary, verbose_name=verbose_name, aggregate=aggregate
        )

        if resolved == "long":
            if favorability:
                prepared["favorability"] = self._to_favorability(
                    prepared["greater_is_better"]
                )
            return prepared.drop(columns="greater_is_better", errors="ignore")

        return squeeze_single_column(
            self._long_to_wide(
                prepared,
                aggregate=aggregate,
                favorability=favorability,
                verbose_name=verbose_name,
                flat_index=flat_index,
            )
        )

    def _repr_html_(self) -> str:
        aggregate = cast(Aggregate, ("mean", "std"))
        frame = self.frame(
            format="auto",
            aggregate=aggregate,
            verbose_name=True,
            flat_index=False,
        )
        html = (
            frame.to_frame()._repr_html_()
            if isinstance(frame, pd.Series)
            else frame._repr_html_()
        )
        return (
            f"{html}"
            '<p role="note">Use <code>.frame()</code> to control the format'
            " of the output.</p>"
        )

    def __repr__(self) -> str:
        aggregate = cast(Aggregate, ("mean", "std"))
        frame = self.frame(
            format="auto",
            aggregate=aggregate,
            verbose_name=False,
            flat_index=True,
        )
        return f"{frame!r}\nUse .frame() to control the format of the output."

    @DisplayMixin.style_plot
    def plot(self) -> Figure:
        """Plot the metrics summary (not implemented)."""
        raise NotImplementedError
