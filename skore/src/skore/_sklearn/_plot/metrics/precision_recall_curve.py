from collections.abc import Sequence
from typing import Any, Literal, cast

import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import LabelBinarizer

from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn._plot.utils import (
    _ClassifierCurveDisplayMixin,
    _despine_matplotlib_axis,
    _validate_style_kwargs,
)
from skore._sklearn.types import (
    DataSource,
    MLTask,
    PositiveLabel,
    ReportType,
    YPlotData,
)


class PrecisionRecallCurveDisplay(_ClassifierCurveDisplayMixin, DisplayMixin):
    """Precision Recall visualization.

    An instance of this class should be created by
    `EstimatorReport.metrics.precision_recall()`. You should not create an
    instance of this class directly.

    Parameters
    ----------
    precision_recall : DataFrame
        The precision-recall curve data to display. The columns are

        - `estimator_name`
        - `split` (may be null)
        - `label`
        - `threshold`
        - `precision`
        - `recall`.

    average_precision : DataFrame
        The average precision data to display. The columns are

        - `estimator_name`
        - `split` (may be null)
        - `label`
        - `average_precision`.

    pos_label : int, float, bool, str or None
        The class considered as the positive class. If None, the class will not
        be shown in the legend.

    data_source : {"train", "test", "X_y", "both"}
        The data source used to compute the precision recall curve.

    ml_task : {"binary-classification", "multiclass-classification"}
        The machine learning task.

    report_type : {"comparison-cross-validation", "comparison-estimator", \
            "cross-validation", "estimator"}
        The type of report.

    Attributes
    ----------
    ax_ : matplotlib axes or ndarray of axes
        The axes on which the precision-recall curve is plotted.

    figure_ : matplotlib figure
        The figure on which the precision-recall curve is plotted.

    lines_ : list of matplotlib lines
        The lines of the precision-recall curve.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skore import train_test_split
    >>> from skore import EstimatorReport
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
    >>> classifier = LogisticRegression(max_iter=10_000)
    >>> report = EstimatorReport(classifier, **split_data)
    >>> display = report.metrics.precision_recall()
    >>> display.plot(relplot_kwargs={"palette": "Set2"})
    """

    _default_relplot_kwargs: dict[str, Any] = {
        "kind": "line",
        "estimator": None,
        "x": "recall",
        "y": "precision",
        "height": 6,
        "aspect": 1,
        "legend": False,
        "facet_kws": {
            "sharex": False,
            "sharey": False,
            "xlim": (-0.01, 1.01),
            "ylim": (-0.01, 1.01),
        },
        "drawstyle": "steps-post",
    }

    def __init__(
        self,
        *,
        precision_recall: DataFrame,
        average_precision: DataFrame,
        pos_label: PositiveLabel | None,
        data_source: DataSource | Literal["both"],
        ml_task: MLTask,
        report_type: ReportType,
    ) -> None:
        self.precision_recall = precision_recall
        self.average_precision = average_precision
        self.pos_label = pos_label
        self.data_source = data_source
        self.ml_task = ml_task
        self.report_type = report_type

    def _get_plot_columns(
        self,
        plot_data: DataFrame,
        subplot_by: str | Literal["auto"] | None = "auto",
    ) -> tuple[str | None, str | None]:
        """Determine subplot (col) and hue columns based on data and user preference.

        Rules:
        - Default ("auto"): "label" for multiclass, None for binary
        - subplot_by=None only allowed for binary classification
        - subplot_by="estimator_name" only allowed for comparison reports
        - subplot_by="label" only allowed for multiclass classification
        - hue priority: estimator_name > label > data_source (excluding col)

        Returns (col, hue) tuple where each can be None if not applicable.
        """
        has_multiple_estimators = (
            "estimator_name" in plot_data.columns
            and plot_data["estimator_name"].nunique() > 1
        )
        is_comparison = self.report_type in (
            "comparison-estimator",
            "comparison-cross-validation",
        )
        is_multiclass = self.ml_task == "multiclass-classification"
        has_both_data_sources = self.data_source == "both"

        allowed_values: set[str | None] = {"auto"}
        if is_multiclass:
            allowed_values.add("label")
        else:
            allowed_values.add(None)
        if is_comparison and has_multiple_estimators:
            allowed_values.add("estimator_name")
        if has_both_data_sources:
            allowed_values.add("data_source")

        if subplot_by not in allowed_values:
            string_values = sorted([v for v in allowed_values if isinstance(v, str)])
            allowed_list = [f"'{v}'" for v in string_values]
            if None in allowed_values:
                allowed_list.append("None")
            allowed_str = ", ".join(allowed_list)
            raise ValueError(
                f"subplot_by must be one of {allowed_str}, got {subplot_by!r} instead."
            )

        if subplot_by == "auto":
            col = "label" if is_multiclass else None
        else:
            col = subplot_by
        has_multiple_labels = (
            "label" in plot_data.columns and plot_data["label"].nunique() > 1
        )
        has_multiple_data_sources = (
            "data_source" in plot_data.columns
            and plot_data["data_source"].nunique() > 1
        )

        hue_candidates = []
        if has_multiple_estimators:
            hue_candidates.append("estimator_name")
        if has_multiple_labels:
            hue_candidates.append("label")
        if has_multiple_data_sources:
            hue_candidates.append("data_source")

        hue = next((c for c in hue_candidates if c != col), None)

        return col, hue

    def _build_legend_for_ax(
        self,
        *,
        ax: Axes,
        stats_df: DataFrame,
        aggregate: bool = False,
    ) -> None:
        """Build custom legend with AP stats for a single axis."""
        if " = " in ax.get_title():
            ax_col, ax_col_value = ax.get_title().split(" = ")[-2:]
            stats_df = stats_df.query(f"{ax_col} == '{ax_col_value}'")

        candidate_cols = ["estimator_name", "label", "data_source"]
        if not aggregate:
            candidate_cols.append("split")

        grouping_cols = [
            c
            for c in candidate_cols
            if c in stats_df.columns and stats_df[c].nunique() > 1
        ]

        new_labels = []
        if not grouping_cols:
            ap_series = stats_df["average_precision"]
            new_labels = [self._format_ap_stat(ap_series, aggregate)]
        elif len(grouping_cols) == 1:
            col_name = grouping_cols[0]
            col = stats_df[col_name]
            order = col.cat.categories if hasattr(col, "cat") else col.unique()
            for val in order:
                mask = col == val
                if mask.any():
                    ap_series = stats_df.loc[mask, "average_precision"]
                    stat_str = self._format_ap_stat(ap_series, aggregate)
                    lbl = self._format_legend_label(col_name, val, stat_str)
                    new_labels.append(lbl)
        else:
            col1_name, col2_name = grouping_cols[0], grouping_cols[1]
            col1 = stats_df[col1_name]
            col2 = stats_df[col2_name]
            order1 = col1.cat.categories if hasattr(col1, "cat") else col1.unique()
            order2 = col2.cat.categories if hasattr(col2, "cat") else col2.unique()
            for val1 in order1:
                for val2 in order2:
                    mask = (col1 == val1) & (col2 == val2)
                    if mask.any():
                        ap_series = stats_df.loc[mask, "average_precision"]
                        stat_str = self._format_ap_stat(ap_series, aggregate)
                        val1_short = self._truncate_name(str(val1))
                        val2_short = self._truncate_name(str(val2))
                        new_labels.append(f"{val1_short} | {val2_short} ({stat_str})")

        if not new_labels:
            return

        n_entries = len(new_labels)

        lines = ax.get_lines()
        if aggregate:
            seen_colors = []
            for line in lines:
                color = line.get_color()
                if color not in seen_colors:
                    seen_colors.append(color)
            handles = [Line2D([0], [0], color=c, lw=2) for c in seen_colors[:n_entries]]
        else:
            handles = lines[:n_entries]

        fontsize = "small" if n_entries > 4 else "medium"

        ax.legend(
            handles,
            new_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=1,
            frameon=False,
            fontsize=fontsize,
        )

    def _truncate_name(self, name: str, max_len: int = 20) -> str:
        """Truncate long names with ellipsis."""
        if len(name) <= max_len:
            return name
        return name[: max_len - 1] + "…"

    def _adjust_figure_for_legend(self, n_legend_rows: int) -> None:
        """Adjust figure bottom margin and height based on expected legend size."""
        legend_height_inches = n_legend_rows * 0.25 + 1
        current_height = self.figure_.get_figheight()

        new_height = current_height + legend_height_inches
        self.figure_.set_figheight(new_height)

        bottom_ratio = legend_height_inches / new_height
        self.figure_.subplots_adjust(bottom=bottom_ratio)

    def _format_ap_stat(self, ap_series: Any, aggregate: bool) -> str:
        """Format AP statistic as single value or mean±std."""
        if aggregate and len(ap_series) > 1:
            return f"AP={ap_series.mean():.2f}±{ap_series.std():.2f}"
        return f"AP={ap_series.iloc[0]:.2f}"

    def _format_legend_label(self, col_name: str, val: Any, stat_str: str) -> str:
        """Format a legend label based on column type."""
        val_str = self._truncate_name(str(val))
        if col_name == "estimator_name":
            return f"{val_str} ({stat_str})"
        elif col_name == "data_source":
            return f"{val_str.title()} set ({stat_str})"
        elif col_name == "label":
            return f"{val_str} ({stat_str})"
        elif col_name == "split":
            return f"Split #{int(val) + 1} ({stat_str})"
        return f"{val_str} ({stat_str})"

    def _plot_with_seaborn(
        self,
        *,
        subplot_by: str | Literal["auto"] | None = "auto",
        relplot_kwargs: dict[str, Any] | None = None,
        estimator_name: str | None = None,
        is_cross_validation: bool = False,
    ) -> None:
        """Unified plotting function using seaborn relplot.

        Parameters
        ----------
        subplot_by : str, "auto", or None
            Column to use for subplots.
        relplot_kwargs : dict, optional
            User-provided kwargs for relplot.
        estimator_name : str, optional
            Estimator name for title (single estimator reports only).
        is_cross_validation : bool
            If True, add units="split" and alpha=0.4, and use aggregate=True for legend.
        """
        plot_data = self.frame(with_average_precision=True)

        col, hue = self._get_plot_columns(plot_data, subplot_by)

        cols_to_convert = ["label", "data_source"]
        if is_cross_validation:
            cols_to_convert.append("split")
        if self.report_type in ("comparison-estimator", "comparison-cross-validation"):
            cols_to_convert.append("estimator_name")

        for c in cols_to_convert:
            if c in plot_data.columns and hasattr(plot_data[c], "cat"):
                plot_data[c] = plot_data[c].astype(str)

        kwargs: dict[str, Any] = {
            "data": plot_data,
            "col": col,
            "hue": hue,
        }

        if is_cross_validation:
            kwargs["units"] = "split"
            kwargs["alpha"] = 0.4

        if self.data_source == "both" and not is_cross_validation:
            kwargs["style"] = "data_source"
            kwargs["dashes"] = {"train": (2, 2), "test": (1, 0)}

        kwargs = _validate_style_kwargs(
            {**kwargs, **self._default_relplot_kwargs},
            relplot_kwargs or {},
        )

        facet_grid = sns.relplot(**kwargs)

        self.figure_ = facet_grid.figure
        self.ax_ = facet_grid.axes.flatten()
        self.lines_ = [line for ax in self.ax_ for line in ax.get_lines()]

        if self.ml_task == "binary-classification":
            info_pos_label = (
                f"\nPositive label: {self.pos_label}"
                if self.pos_label is not None
                else ""
            )
        else:
            info_pos_label = ""

        info_data_source = (
            f"\nData source: {self.data_source.capitalize()} set"
            if self.data_source in ("train", "test")
            else "\nData source: external set"
            if self.data_source != "both"
            else ""
        )

        aggregate = is_cross_validation
        for ax in self.ax_:
            self._build_legend_for_ax(ax=ax, stats_df=plot_data, aggregate=aggregate)

        self._adjust_figure_for_legend(plot_data[hue].nunique() if hue else 1)

        title = (
            f"Precision-Recall Curve for {estimator_name}"
            if estimator_name
            else "Precision-Recall Curve"
        )
        facet_grid.figure.suptitle(
            title + info_pos_label + info_data_source,
            y=1.02,
        )

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        estimator_name: str | None = None,
        subplot_by: str | Literal["auto"] | None = "auto",
        relplot_kwargs: dict[str, Any] | None = None,
        despine: bool = True,
    ) -> None:
        """Plot visualization.

        Parameters
        ----------
        estimator_name : str, default=None
            Name of the estimator used to plot the precision-recall curve. If
            `None`, we use the inferred name from the estimator.

        subplot_by : str, "auto", or None, default="auto"
            Column to use for creating subplots. Options:
            - "auto": "label" for multiclass, None for binary
            - "label": one subplot per class (multiclass only)
            - "estimator_name": one subplot per estimator (comparison only)
            - None: no subplots (binary only)
            Note: "split" is not allowed.

        relplot_kwargs : dict, default=None
            Keyword arguments to be passed to seaborn's `relplot` for rendering
            the precision-recall curve(s). Common options include `palette`,
            `alpha`, `linewidth`, etc.

        despine : bool, default=True
            Whether to remove the top and right spines from the plot.

        Notes
        -----
        The average precision (cf. :func:`~sklearn.metrics.average_precision_score`)
        in scikit-learn is computed without any interpolation. To be consistent
        with this metric, the precision-recall curve is plotted without any
        interpolation as well (step-wise style).

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(classifier, **split_data)
        >>> display = report.metrics.precision_recall()
        >>> display.plot(relplot_kwargs={"palette": "Set2", "alpha": 0.8})
        """
        return self._plot(
            estimator_name=estimator_name,
            subplot_by=subplot_by,
            relplot_kwargs=relplot_kwargs,
            despine=despine,
        )

    def _plot_matplotlib(
        self,
        *,
        estimator_name: str | None = None,
        subplot_by: str | Literal["auto"] | None = "auto",
        relplot_kwargs: dict[str, Any] | None = None,
        despine: bool = True,
    ) -> None:
        """Matplotlib implementation of the `plot` method."""
        is_cross_validation = self.report_type in (
            "cross-validation",
            "comparison-cross-validation",
        )

        # Get estimator name for single estimator reports
        plot_estimator_name = None
        if self.report_type in ("estimator", "cross-validation"):
            plot_estimator_name = (
                self.precision_recall["estimator_name"].cat.categories.item()
                if estimator_name is None
                else estimator_name
            )

        self._plot_with_seaborn(
            subplot_by=subplot_by,
            relplot_kwargs=relplot_kwargs,
            estimator_name=plot_estimator_name,
            is_cross_validation=is_cross_validation,
        )

        if despine:
            for ax in self.ax_:
                _despine_matplotlib_axis(ax)

    @classmethod
    def _compute_data_for_display(
        cls,
        y_true: Sequence[YPlotData],
        y_pred: Sequence[YPlotData],
        *,
        report_type: ReportType,
        estimators: Sequence[BaseEstimator],
        ml_task: MLTask,
        data_source: DataSource | Literal["both"],
        pos_label: PositiveLabel | None,
        drop_intermediate: bool = True,
    ) -> "PrecisionRecallCurveDisplay":
        """Plot precision-recall curve given binary class predictions.

        Parameters
        ----------
        y_true : list of array-like of shape (n_samples,)
            True binary labels.

        y_pred : list of array-like of shape (n_samples,)
            Target scores, can either be probability estimates of the positive class,
            confidence values, or non-thresholded measure of decisions (as returned by
            "decision_function" on some classifiers).

        report_type : {"comparison-cross-validation", "comparison-estimator", \
                "cross-validation", "estimator"}
            The type of report.

        estimators : list of estimator instances
            The estimators from which `y_pred` is obtained.

        ml_task : {"binary-classification", "multiclass-classification"}
            The machine learning task.

        data_source : {"train", "test", "X_y", "both"}
            The data source used to compute the precision recall curve.

        pos_label : int, float, bool, str or none
            The class considered as the positive class when computing the
            precision and recall metrics.

        drop_intermediate : bool, default=True
            Whether to drop some suboptimal thresholds which would not appear
            on a plotted precision-recall curve. This is useful in order to
            create lighter precision-recall curves.

        Returns
        -------
        display : PrecisionRecallCurveDisplay
        """
        pos_label_validated = cls._validate_from_predictions_params(
            y_true, y_pred, ml_task=ml_task, pos_label=pos_label
        )

        precision_recall_records = []
        average_precision_records = []

        if ml_task == "binary-classification":
            for y_true_i, y_pred_i in zip(y_true, y_pred, strict=False):
                pos_label_validated = cast(PositiveLabel, pos_label_validated)
                precision_i, recall_i, thresholds_i = precision_recall_curve(
                    y_true_i.y,
                    y_pred_i.y,
                    pos_label=pos_label_validated,
                    drop_intermediate=drop_intermediate,
                )
                average_precision_i = average_precision_score(
                    y_true_i.y, y_pred_i.y, pos_label=pos_label_validated
                )

                for precision, recall, threshold in zip(
                    precision_i, recall_i, thresholds_i, strict=False
                ):
                    precision_recall_records.append(
                        {
                            "estimator_name": y_true_i.estimator_name,
                            "data_source": y_true_i.data_source,
                            "split": y_true_i.split,
                            "label": pos_label_validated,
                            "threshold": threshold,
                            "precision": precision,
                            "recall": recall,
                        }
                    )
                average_precision_records.append(
                    {
                        "estimator_name": y_true_i.estimator_name,
                        "data_source": y_true_i.data_source,
                        "split": y_true_i.split,
                        "label": pos_label_validated,
                        "average_precision": average_precision_i,
                    }
                )
        else:  # multiclass-classification
            classes = estimators[0].classes_
            for y_true_i, y_pred_i in zip(y_true, y_pred, strict=True):
                label_binarizer = LabelBinarizer().fit(classes)
                y_true_onehot_i: NDArray = label_binarizer.transform(y_true_i.y)
                y_pred_i_y = cast(NDArray, y_pred_i.y)

                for class_idx, class_ in enumerate(classes):
                    precision_class_i, recall_class_i, thresholds_class_i = (
                        precision_recall_curve(
                            y_true_onehot_i[:, class_idx],
                            y_pred_i_y[:, class_idx],
                            pos_label=None,
                            drop_intermediate=drop_intermediate,
                        )
                    )
                    average_precision_class_i = average_precision_score(
                        y_true_onehot_i[:, class_idx], y_pred_i_y[:, class_idx]
                    )

                    for precision, recall, threshold in zip(
                        precision_class_i,
                        recall_class_i,
                        thresholds_class_i,
                        strict=False,
                    ):
                        precision_recall_records.append(
                            {
                                "estimator_name": y_true_i.estimator_name,
                                "data_source": y_true_i.data_source,
                                "split": y_true_i.split,
                                "label": class_,
                                "threshold": threshold,
                                "precision": precision,
                                "recall": recall,
                            }
                        )
                    average_precision_records.append(
                        {
                            "estimator_name": y_true_i.estimator_name,
                            "data_source": y_true_i.data_source,
                            "split": y_true_i.split,
                            "label": class_,
                            "average_precision": average_precision_class_i,
                        }
                    )

        dtypes = {
            "estimator_name": "category",
            "data_source": "category",
            "split": "category",
            "label": "category",
        }

        return cls(
            precision_recall=DataFrame.from_records(precision_recall_records).astype(
                dtypes
            ),
            average_precision=DataFrame.from_records(average_precision_records).astype(
                dtypes
            ),
            pos_label=pos_label_validated,
            data_source=data_source,
            ml_task=ml_task,
            report_type=report_type,
        )

    def frame(self, with_average_precision: bool = False) -> DataFrame:
        """Get the data used to create the precision-recall curve plot.

        Parameters
        ----------
        with_average_precision : bool, default=False
            Whether to include the average precision column in the returned DataFrame.

        Returns
        -------
        DataFrame
            A DataFrame containing the precision-recall curve data with columns
            depending on the report type:

            - `estimator_name`: Name of the estimator (when comparing estimators)
            - `split`: Cross-validation split ID (when doing cross-validation)
            - `label`: Class label (for multiclass-classification)
            - `threshold`: Decision threshold
            - `precision`: Precision score at threshold
            - `recall`: Recall score at threshold
            - `average_precision`: average precision
              (when `with_average_precision=True`)

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split, EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> clf = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(clf, **split_data)
        >>> display = report.metrics.precision_recall()
        >>> df = display.frame()
        """
        if with_average_precision:
            # The merge between the precision-recall curve and the average precision is
            # done without specifying the columns to merge on, hence done on all column
            # that are present in both DataFrames.
            # In this case, the common columns are all columns but not the ones
            # containing the statistics.
            df = self.precision_recall.merge(self.average_precision)
        else:
            df = self.precision_recall

        statistical_columns = ["threshold", "precision", "recall"]
        if with_average_precision:
            statistical_columns.append("average_precision")

        if self.report_type == "estimator":
            indexing_columns = []
        elif self.report_type == "cross-validation":
            indexing_columns = ["split"]
        elif self.report_type == "comparison-estimator":
            indexing_columns = ["estimator_name"]
        else:  # self.report_type == "comparison-cross-validation"
            indexing_columns = ["estimator_name", "split"]

        if self.data_source == "both":
            indexing_columns += ["data_source"]

        if self.ml_task == "binary-classification":
            columns = indexing_columns + statistical_columns
        else:
            columns = indexing_columns + ["label"] + statistical_columns

        return df[columns]
