from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any, Literal, NotRequired, TypedDict, cast

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.utils.validation import _is_arraylike

from skore._sklearn._plot.base import DisplayMixin
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
PIVOT_VALUE_COLUMN = "value"
PIVOT_META_COLUMN = "greater_is_better"


def _resolve_pivot_index_keys(df: pd.DataFrame) -> list[str]:
    """Return row-index keys present in ``df``."""
    return [
        col
        for col in METRIC_INDEX_KEYS
        if col == "metric" or (col in df.columns and df[col].notna().any())
    ]


def _resolve_pivot_column_keys(
    report_type: ReportType,
    df: pd.DataFrame,
    *,
    aggregate: Aggregate | None,
) -> list[str]:
    """Return pivot column keys for ``df``, mirroring roc_curve indexing columns.

    When ``aggregate`` is set, cross-validation splits are reduced with an
    aggregation function during the pivot, so ``split`` is not a pivot column.
    """
    has_both_sources = "data_source" in df.columns and df["data_source"].nunique() > 1
    keys: list[MetricColumnKey]
    if report_type == "estimator":
        keys = ["data_source"] if has_both_sources else []
    elif report_type == "comparison-estimator":
        keys = ["estimator", "data_source"] if has_both_sources else ["estimator"]
    elif report_type == "cross-validation":
        keys = [] if aggregate is not None else ["split"]
    else:  # comparison-cross-validation
        keys = ["estimator"] if aggregate is not None else ["estimator", "split"]
    return [key for key in keys if key in df.columns]


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
    estimator: str
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
        ``estimator``, ``data_source``, ``label``, ``output``, ``average``,
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
        errors: list[tuple[Metric, Exception]] | None = None,
    ):
        self.summary = summary
        self.report_type = report_type
        # Remove duplicates while preserving order
        # Use repr because Metrics and Exceptions are not comparable
        self.errors = list({repr(x): x for x in (errors or [])}.values())

    @classmethod
    def _compute_data_for_display(
        cls,
        rows: list[MetricsSummaryRow],
        *,
        report_type: ReportType,
        errors: list[tuple[Metric, Exception]] | None = None,
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

    def _prepare_long(
        self,
        summary: pd.DataFrame,
        *,
        verbose_name: bool = False,
    ) -> pd.DataFrame:
        """Select the columns needed for pivoting and normalize their names."""
        metric_col = "verbose_name" if verbose_name else "name"

        columns: list[str] = []
        if "comparison" in self.report_type:
            columns.append("estimator")
        if "cross-validation" in self.report_type:
            columns.append("split")
        if summary["data_source"].nunique() > 1:
            columns.append("data_source")
        columns.append(metric_col)
        columns.extend(
            col for col in METRIC_DIMENSION_KEYS if summary[col].notna().any()
        )
        columns.extend(["score", "greater_is_better"])

        frame = summary[columns].copy()
        return frame.rename(
            columns={
                metric_col: "metric",
                "score": "value",
            }
        ).reset_index(drop=True)

    def _pivot_estimator(
        self,
        df: pd.DataFrame,
        index_cols: list[str],
        column_cols: list[str],
    ) -> pd.DataFrame:
        """Pivot a single-estimator table."""
        estimator = self.summary["estimator"].iloc[0]
        if not column_cols:
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
        index_cols: list[str],
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
            table.columns = pd.MultiIndex.from_product(
                [[estimator], [f"Split #{col}" for col in table.columns]]
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
        index_cols: list[str],
        *,
        aggregate: Aggregate | None,
    ) -> pd.DataFrame:
        """Pivot cross-validation metrics."""
        estimator = self.summary["estimator"].iloc[0]
        if "data_source" in df.columns and df["data_source"].nunique() > 1:
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
        index_cols: list[str],
        column_cols: list[str],
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
        index_cols: list[str],
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
        index_cols: list[str],
        *,
        favorability: bool,
        verbose_name: bool,
        flat_index: bool,
    ) -> pd.DataFrame:
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
        return table

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

        prepared = self._prepare_long(
            self._resolve_fingerprints(self.summary),
            verbose_name=verbose_name,
        )
        index_cols = _resolve_pivot_index_keys(prepared)
        column_cols = _resolve_pivot_column_keys(
            self.report_type, prepared, aggregate=aggregate
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

        table = self._finalize(
            table,
            prepared,
            index_cols,
            favorability=favorability,
            verbose_name=verbose_name,
            flat_index=flat_index,
        )
        return squeeze_single_column(table, lowercase=not verbose_name)

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

    @DisplayMixin.style_plot
    def plot(self) -> Figure:
        """Plot the metrics summary (not implemented)."""
        raise NotImplementedError()
