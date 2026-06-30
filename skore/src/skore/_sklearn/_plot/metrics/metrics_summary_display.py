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
from skore._utils._index import flatten_multi_index, maybe_squeeze_single_column

FrameFormat = Literal["long", "wide", "auto"]

_FAVORABILITY_SYMBOLS = {True: "(↗︎)", False: "(↘︎)"}


def frame_repr_html(frame: pd.DataFrame | pd.Series) -> str:
    """Return the HTML representation of a metrics summary frame or series."""
    if isinstance(frame, pd.Series):
        return frame.to_frame()._repr_html_()
    return frame._repr_html_()


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

    Use :meth:`frame` to export the summary as a :class:`pandas.DataFrame` or,
    for wide layouts with a single value column, as a named
    :class:`pandas.Series`.

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
        Always includes a ``fingerprint`` column; it is resolved and omitted from
        :meth:`frame` output.
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

            def average_to_string(value: Any) -> Any:
                if _is_arraylike(value):
                    return str(np.asarray(value).tolist())
                return value

            summary["average"] = (
                summary["average"].map(average_to_string).astype("string")
            )

        return cls(summary, report_type=report_type)

    @staticmethod
    def _resolve_fingerprints(data: pd.DataFrame) -> pd.DataFrame:
        """Disambiguate metric labels across distinct fingerprints.

        When several rows share a label (the technical ``name`` or its
        ``verbose_name``) but come from metrics with different fingerprints,
        they are renamed ``{label}_1``, ``{label}_2``, ... in the order they
        first appear. ``None`` counts as a regular fingerprint value, so a
        custom metric reusing a built-in's name will still get disambiguated
        against the built-in.

        Both ``name`` and ``verbose_name`` are disambiguated independently, so
        whichever column :meth:`frame` displays is unambiguous.

        Suffixes skip over any label already present in the column, so we don't
        produce a collision if a metric happens to already be called e.g.
        ``"Metric_1"``.
        """
        data = data.copy()

        for label_col in ("name", "verbose_name"):
            fingerprints_per_label = defaultdict(list)
            for label, fingerprint in (
                data[[label_col, "fingerprint"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            ):
                fingerprints_per_label[label].append(fingerprint)

            renaming = {}
            existing = set(data[label_col])
            for label, fingerprints in fingerprints_per_label.items():
                if len(fingerprints) < 2:
                    continue
                i = 0
                for fingerprint in fingerprints:
                    while True:
                        i += 1
                        candidate = f"{label}_{i}"
                        if candidate not in existing:
                            break
                    existing.add(candidate)
                    renaming[(label, fingerprint)] = candidate

            for (label, fingerprint), new_name in renaming.items():
                fp_match = (
                    data["fingerprint"].isna()
                    if pd.isna(fingerprint)
                    else data["fingerprint"] == fingerprint
                )
                data.loc[(data[label_col] == label) & fp_match, label_col] = new_name

        return data

    def _prepare_pivot_df(
        self, prepared: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Return a pivot-ready frame and the columns used as the row index.

        ``metric`` is always part of the index; the metric dimensions
        (``label``, ``output``, ``average``) join it only when they carry a
        value. ``.summary`` stores their missing entries as ``pd.NA``, but
        pandas pivots drop or reorder ``pd.NA`` index keys, so they are filled
        with an empty-string sentinel here (and rendered as empty at the
        display boundary). Integer labels keep their ``Int64`` values, so no
        float coercion is needed.
        """
        index_cols = ["metric"] + [
            col
            for col in ("label", "output", "average")
            if col in prepared.columns and prepared[col].notna().any()
        ]
        pivot_df = prepared.copy()
        for col in ("label", "output", "average"):
            if col in pivot_df.columns:
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

    def _pivot_to_wide(
        self,
        prepared: pd.DataFrame,
        *,
        aggregate: Aggregate | None = ("mean", "std"),
        favorability: bool = False,
        verbose_name: bool = False,
        with_multiindex: bool = False,
    ) -> pd.DataFrame:
        """Pivot a prepared long table into wide layout."""
        if self.report_type == "estimator":
            df = self._pivot_estimator_wide(prepared)
        elif self.report_type == "cross-validation":
            df = self._pivot_cross_validation_wide(prepared, aggregate=aggregate)
        elif self.report_type == "comparison-estimator":
            df = self._pivot_comparison_estimator_wide(prepared)
        else:
            df = self._pivot_comparison_cross_validation_wide(
                prepared, aggregate=aggregate
            )

        if favorability:
            # ``greater_is_better`` is constant per metric row (it does not vary
            # with ``data_source``, ``split``, ``aggregate`` or ``estimator``), so
            # a single ``groupby(index_cols).first()`` yields the value for every
            # wide layout.
            pivot_df, index_cols = self._prepare_pivot_df(prepared)
            favorability_col = self._to_favorability(
                pivot_df.groupby(index_cols, dropna=False)["greater_is_better"].first()
            )
            favorability_col.index = favorability_col.index.set_names(df.index.names)
            df["Favorability"] = favorability_col

        if not with_multiindex:
            lowercase = not verbose_name
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = flatten_multi_index(df.columns, lowercase=lowercase)
            if isinstance(df.index, pd.MultiIndex):
                df.index = flatten_multi_index(df.index, lowercase=lowercase)
            if isinstance(df.index, pd.Index):
                df.index = df.index.str.replace(r"\((.*)\)$", r"\1", regex=True)
        else:
            if isinstance(df.index, pd.MultiIndex):
                levels = list(df.index.levels)
                for level_index, name in enumerate(df.index.names):
                    if name == "Label":
                        levels[level_index] = pd.Index(
                            [
                                "" if value == "" else str(value)
                                for value in levels[level_index]
                            ],
                            dtype="string",
                            name=name,
                        )
                df.index = df.index.set_levels(levels)

        return df

    def _pivot_estimator_wide(
        self,
        prepared: pd.DataFrame,
        *,
        estimator_name: str | None = None,
    ) -> pd.DataFrame:
        """Pivot an estimator report into wide layout (one value column)."""
        pivot_df, index_cols = self._prepare_pivot_df(prepared)
        if estimator_name is None:
            estimator_name = self.summary["estimator_name"].iloc[0]

        if (
            "data_source" not in prepared.columns
            or prepared["data_source"].nunique() == 1
        ):
            df = pivot_df.set_index(index_cols)[["value"]]
            df.columns = [estimator_name]
        else:
            df = pivot_df.pivot_table(
                index=index_cols,
                columns="data_source",
                values="value",
                aggfunc="first",
                sort=False,
            )
            df = df[["train", "test"]]
            df.columns = [f"{estimator_name} ({col})" for col in df.columns]

        df.index = df.index.set_names(
            ["Metric", "Label", "Output", "Average"][: df.index.nlevels]
        )
        return df

    def _pivot_cross_validation_wide(
        self,
        prepared: pd.DataFrame,
        *,
        aggregate: Aggregate | None,
        estimator_name: str | None = None,
    ) -> pd.DataFrame:
        """Pivot a cross-validation report into wide layout (split/aggregate cols)."""
        if "data_source" in prepared.columns and prepared["data_source"].nunique() > 1:
            frames = []
            for data_source in ("train", "test"):
                source_df = prepared[prepared["data_source"] == data_source]
                source_frame = self._pivot_cross_validation_wide(
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

        pivot_df, index_cols = self._prepare_pivot_df(prepared)
        if estimator_name is None:
            estimator_name = self.summary["estimator_name"].iloc[0]

        if "aggregate" in prepared.columns:
            df = pivot_df.pivot_table(
                index=index_cols,
                columns="aggregate",
                values="value",
                aggfunc="first",
                sort=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.swaplevel(0, 1)
            elif isinstance(aggregate, str):
                df.columns = pd.MultiIndex.from_tuples([(estimator_name, aggregate)])
            else:
                df.columns = pd.MultiIndex.from_tuples(
                    [(estimator_name, str(col)) for col in df.columns]
                )
        else:
            df = pivot_df.pivot_table(
                index=index_cols,
                columns="split",
                values="value",
                aggfunc="first",
                sort=False,
            )
            df.columns = pd.MultiIndex.from_product(
                [[estimator_name], [f"Split #{i}" for i in df.columns]]
            )

        df.index = df.index.set_names(
            ["Metric", "Label", "Output", "Average"][: df.index.nlevels]
        )
        return df

    def _pivot_comparison_estimator_wide(
        self,
        prepared: pd.DataFrame,
    ) -> pd.DataFrame:
        """Pivot a comparison-estimator report into wide layout (one col per est.)."""
        df = pd.concat(
            [
                self._pivot_estimator_wide(
                    estimator_df.drop(columns="estimator"),
                    estimator_name=estimator_key,
                )
                for estimator_key, estimator_df in prepared.groupby(
                    "estimator", sort=False
                )
            ],
            axis="columns",
        )
        df.columns.name = "Estimator"
        return df

    def _pivot_comparison_cross_validation_wide(
        self,
        prepared: pd.DataFrame,
        *,
        aggregate: Aggregate | None,
    ) -> pd.DataFrame:
        """Pivot a comparison-cross-validation report into wide layout."""
        estimators: list[str] = []
        frames = []
        for estimator_key, estimator_df in prepared.groupby("estimator", sort=False):
            estimators.append(estimator_key)
            frames.append(
                self._pivot_cross_validation_wide(
                    estimator_df.drop(columns="estimator"),
                    aggregate=aggregate,
                    estimator_name=estimator_key,
                )
            )

        df = pd.concat(frames, axis="columns")

        if aggregate is None:
            df.columns.names = ["Estimator", "Split"]
        else:
            # Group the aggregate level (e.g. mean, std) together while keeping
            # estimators in the order they were passed to the comparison.
            df = df.swaplevel(0, 1, axis="columns")
            aggregate_order = list(dict.fromkeys(df.columns.get_level_values(0)))
            df = df[
                [
                    (agg, estimator)
                    for agg in aggregate_order
                    for estimator in estimators
                ]
            ]
            df.columns.names = [None, "Estimator"]

        return df

    def _repr_frame(self, *, for_html: bool = False) -> pd.DataFrame | pd.Series:
        """Return the dataframe used for display representation."""
        aggregate = cast(Aggregate, ("mean", "std"))
        return self.frame(
            format="auto",
            aggregate=aggregate,
            verbose_name=for_html,
            with_multiindex=for_html,
        )

    def frame(
        self,
        *,
        format: FrameFormat = "auto",
        favorability: bool = False,
        verbose_name: bool = False,
        with_multiindex: bool = False,
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

        with_multiindex : bool, default=False
            Whether to keep the row and column MultiIndex in the ``"wide"``
            layout. When ``False``, the index and columns are flattened into a
            single level. Has no effect on the ``"long"`` layout.

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

        Raises
        ------
        ValueError
            If ``format`` is not one of ``"long"``, ``"wide"``, or ``"auto"``.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> clf = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(clf, X, y, splitter=0.2)
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

        return maybe_squeeze_single_column(
            self._pivot_to_wide(
                prepared,
                aggregate=aggregate,
                favorability=favorability,
                verbose_name=verbose_name,
                with_multiindex=with_multiindex,
            )
        )

    def _frame_repr_html(self, frame: pd.DataFrame | pd.Series) -> str:
        return frame_repr_html(frame)

    def _repr_html_(self) -> str:
        return (
            f"{self._frame_repr_html(self._repr_frame(for_html=True))}"
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
