from collections.abc import Sequence
from typing import Any, Literal, cast

import seaborn as sns
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, precision_recall_curve

from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn._plot.utils import (
    _build_custom_legend_with_stats,
    _check_label,
    _ClassifierDisplayMixin,
    _concat_frames_with_column_data,
    _despine_matplotlib_axis,
    _downsample_thresholds_indices,
    _get_curve_plot_columns,
    _one_hot_encode,
    _reorder_categoricals_by_appearance,
    _validate_style_kwargs,
)
from skore._sklearn.types import (
    _DEFAULT,
    DataSource,
    MLTask,
    PositiveLabel,
    ReportType,
)


class PrecisionRecallCurveDisplay(_ClassifierDisplayMixin, DisplayMixin):
    """Plot the precision-recall curve.

    Parameters
    ----------
    precision_recall : DataFrame
        The precision-recall curve data to display. The columns are

        - `estimator`
        - `split` (may be null)
        - `label`
        - `threshold`
        - `precision`
        - `recall`
        - `data_source`.

    average_precision : DataFrame
        The average precision data to display. The columns are

        - `estimator`
        - `split` (may be null)
        - `label`
        - `average_precision`.

    report_pos_label : int, float, bool, str or None
        Default positive label used when `plot(label=...)` is not specified.

    data_source : {"train", "test", "both"}
        The data source used to compute the precision recall curve.

    ml_task : {"binary-classification", "multiclass-classification"}
        The machine learning task.

    report_type : {"comparison-cross-validation", "comparison-estimator", \
            "cross-validation", "estimator"}
        The type of report.

    Attributes
    ----------
    precision_recall : DataFrame
        Precision-recall curve points per threshold.
    average_precision : DataFrame
        Average precision values per label and grouping column.
    report_pos_label : PositiveLabel or None
        Default positive label for plotting.
    data_source : DataSource or "both"
        The data source used to compute the curve.
    ml_task : MLTask
        The machine learning task.
    report_type : ReportType
        The type of report.
    labels : list
        Available class labels.

    See Also
    --------
    EstimatorReport.metrics.precision_recall : Create this display from a report.
    RocCurveDisplay : Plot ROC curves.
    ConfusionMatrixDisplay : Display the confusion matrix.

    Notes
    -----
    For multiclass classification, curves are computed in a one-vs-rest
    fashion for each class label.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skore import evaluate
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> classifier = LogisticRegression(max_iter=10_000)
    >>> report = evaluate(classifier, X, y, splitter=0.2)
    >>> display = report.metrics.precision_recall()
    >>> display.set_style(relplot_kwargs={"palette": "Set2"})
    >>> display.plot()
    """

    _default_relplot_kwargs: dict[str, Any] = {
        "height": 6,
        "aspect": 1,
        "facet_kws": {
            "sharex": False,
            "sharey": False,
            "xlim": (-0.01, 1.01),
            "ylim": (-0.01, 1.01),
        },
        "drawstyle": "steps-post",
        "legend": False,
    }

    def __init__(
        self,
        *,
        precision_recall: DataFrame,
        average_precision: DataFrame,
        report_pos_label: PositiveLabel | None,
        data_source: DataSource | Literal["both"],
        ml_task: MLTask,
        report_type: ReportType,
    ) -> None:
        self.precision_recall = precision_recall
        self.average_precision = average_precision
        self.report_pos_label = report_pos_label
        self.data_source = data_source
        self.ml_task = ml_task
        self.report_type = report_type

    @property
    def labels(self):
        return self.precision_recall["label"].cat.categories.to_list()

    @classmethod
    def _concatenate(
        cls,
        child_displays: Sequence["PrecisionRecallCurveDisplay"],
        *,
        report_type: ReportType,
        data_source: None | Literal["both"] = None,
        column_data: dict[str, list] | None = None,
    ) -> "PrecisionRecallCurveDisplay":
        """Build a precision-recall display by concatenating child displays."""
        first_display = child_displays[0]
        return cls(
            precision_recall=_concat_frames_with_column_data(
                [display.precision_recall for display in child_displays],
                column_data,
            ),
            average_precision=_concat_frames_with_column_data(
                [display.average_precision for display in child_displays],
                column_data,
            ),
            report_pos_label=first_display.report_pos_label,
            data_source=data_source or first_display.data_source,
            ml_task=first_display.ml_task,
            report_type=report_type,
        )

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        subplot_by: (
            Literal["auto", "label", "estimator", "data_source"] | None
        ) = "auto",
        despine: bool = True,
        label: PositiveLabel = _DEFAULT,
    ) -> Figure:
        """Plot visualization.

        Parameters
        ----------
        subplot_by : {"auto", "label", "estimator", "data_source"} or None, \
                default="auto"
            Column to use for creating subplots. Options:

            - "auto": None for EstimatorReport and Cross-Validation Report,
              "estimator" for ComparisonReport
            - "label": one subplot per class when plotting one-vs-rest curves
            - "estimator": one subplot per estimator (comparison only)
            - "data_source": one subplot per data source (EstimatorReport with both
              data sources only)
            - None: no subplots (Not available for comparison in classification
              with no specified label)

        despine : bool, default=True
            Whether to remove the top and right spines from the plot.

        label : int, float, bool, str or None, default=report pos_label
            The class whose curve to select. We always compute one curve per class,
            in a one-vs-rest fashion in multiclass classification and by alternating
            the positive class in binary classification. This lets you display only
            the curve of the desired class. Use `None` to show them all.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the precision-recall curve.

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
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=0.2)
        >>> display = report.metrics.precision_recall()
        >>> display.set_style(relplot_kwargs={"palette": "Set2", "alpha": 0.8})
        >>> display.plot()
        """
        label = _check_label(self.labels, label, self.report_pos_label)
        return self._plot(subplot_by=subplot_by, despine=despine, label=label)

    def _plot_matplotlib(
        self,
        *,
        subplot_by: Literal["auto", "label", "estimator", "data_source"] | None,
        despine: bool,
        label: PositiveLabel,
    ) -> Figure:
        """Matplotlib implementation of the `plot` method."""
        plot_data = self.frame(label=label, with_average_precision=True)

        col, hue, style = _get_curve_plot_columns(
            plot_data=plot_data,
            report_type=self.report_type,
            label=label,
            data_source=self.data_source,
            subplot_by=subplot_by,
        )

        plot_data = _reorder_categoricals_by_appearance(plot_data, [col, hue, style])

        relplot_kwargs: dict[str, Any] = {
            "col": col,
            "hue": hue,
            "style": style,
        }

        if "cross-validation" in self.report_type:
            relplot_kwargs["units"] = "split"
            relplot_kwargs["alpha"] = 0.4

            # Convert the "split" column from category to string to avoid pandas future
            # warning. See: https://github.com/mwaskom/seaborn/issues/3891
            plot_data["split"] = plot_data["split"].astype(str)

        relplot_kwargs["col_order"] = plot_data[col].unique().tolist() if col else None
        relplot_kwargs["hue_order"] = plot_data[hue].unique().tolist() if hue else None
        relplot_kwargs["style_order"] = (
            plot_data[style].unique().tolist() if style else None
        )

        if style:
            relplot_kwargs["dashes"] = {"train": (5, 5), "test": ""}

        facet = sns.relplot(
            data=plot_data,
            kind="line",
            estimator=None,
            x="recall",
            y="precision",
            **_validate_style_kwargs(relplot_kwargs, self._default_relplot_kwargs),
        )

        figure, axes = facet.figure, facet.axes.flatten()

        # Create space under the plot to fit the manually created legends.
        n_legend_rows = plot_data[hue].nunique() if hue else 1
        legend_height_inches = n_legend_rows * 0.25 + 1
        current_height = figure.get_figheight()
        new_height = current_height + legend_height_inches
        figure.set_figheight(new_height)

        # Build a legend for each subplot.
        for idx, ax in enumerate(axes):
            col_value = (
                relplot_kwargs["col_order"][idx]
                if relplot_kwargs["col_order"]
                else None
            )
            subplot_data = plot_data[plot_data[col] == col_value] if col else plot_data
            _build_custom_legend_with_stats(
                ax=ax,
                subplot_data=subplot_data,
                hue=hue,
                style=style,
                hue_order=relplot_kwargs["hue_order"],
                style_order=relplot_kwargs["style_order"],
                is_cross_validation="cross-validation" in self.report_type,
                statistic_column_name="average_precision",
                statistic_acronym="AP",
            )

        if label is not None:
            info_label = (
                f"Positive label: {label}"
                if self.ml_task == "binary-classification"
                else f"Label: {label}"
            )
        else:
            info_label = None

        info_data_source = (
            f"Data source: {self.data_source.capitalize()} set"
            if self.data_source in ("train", "test")
            else None
        )

        title = "Precision-Recall Curve"
        if "comparison" not in self.report_type:
            title += f" for {self.precision_recall['estimator'].cat.categories.item()}"
        figure.suptitle("\n".join(filter(None, [title, info_label, info_data_source])))

        if despine:
            for ax in axes:
                _despine_matplotlib_axis(ax)

        return figure

    @classmethod
    def _compute_data_for_display(
        cls,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        report_type: ReportType,
        estimator: BaseEstimator,
        estimator_name: str,
        ml_task: MLTask,
        data_source: DataSource,
        report_pos_label: PositiveLabel = None,
        drop_intermediate: bool = True,
        max_n_thresholds: int | None = 500,
    ) -> "PrecisionRecallCurveDisplay":
        """Plot precision-recall curve given binary class predictions.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels.

        y_pred : array-like of shape (n_samples, n_classes)
            Target scores, can either be probability estimates or non-thresholded
            measure of decisions (as returned by "decision_function").

        report_type : {"comparison-cross-validation", "comparison-estimator", \
                "cross-validation", "estimator"}
            The type of report.

        estimator : estimator instance
            The estimator from which `y_pred` is obtained.

        estimator_name : str
            The estimator name to attach to the display data.

        ml_task : {"binary-classification", "multiclass-classification"}
            The machine learning task.

        data_source : {"train", "test"}
            The data source used to compute the precision recall curve.

        drop_intermediate : bool, default=True
            Whether to drop some suboptimal thresholds which would not appear
            on a plotted precision-recall curve. This is useful in order to
            create lighter precision-recall curves.

        max_n_thresholds : int or None, default=500
            Cap on the number of thresholds kept per class after the curve is
            computed. When the number of thresholds returned by scikit-learn
            exceeds ``max_n_thresholds``, the curve is downsampled by picking
            evenly-spaced indices from the sorted thresholds (quantile-based
            sampling that preserves the empirical threshold distribution).
            Endpoints are always kept and no interpolation is performed.
            ``None`` disables downsampling. Must be at least 2.

        Returns
        -------
        display : PrecisionRecallCurveDisplay
        """
        classes = estimator.classes_
        y_true_onehot = _one_hot_encode(y_true, classes)
        y_pred_arr = cast(NDArray, y_pred)

        curve_dfs = []
        ap_dfs = []
        for class_idx, label in enumerate(classes):
            curve_df, ap_df = cls._compute_data_ovr(
                y_true=y_true_onehot[:, class_idx],
                y_pred=y_pred_arr[:, class_idx],
                drop_intermediate=drop_intermediate,
                max_n_thresholds=max_n_thresholds,
                # metadata:
                estimator=estimator_name,
                data_source=data_source,
                split=None,
                label=label,
            )
            curve_dfs.append(curve_df)
            ap_dfs.append(ap_df)

        return cls(
            precision_recall=_concat_frames_with_column_data(curve_dfs),
            average_precision=_concat_frames_with_column_data(ap_dfs),
            report_pos_label=report_pos_label,
            data_source=data_source,
            ml_task=ml_task,
            report_type=report_type,
        )

    @staticmethod
    def _compute_data_ovr(
        y_true, y_pred, drop_intermediate, max_n_thresholds, **metadata
    ):
        precision, recall, thresholds = precision_recall_curve(
            y_true,
            y_pred,
            pos_label=1,
            drop_intermediate=drop_intermediate,
        )
        average_precision = average_precision_score(y_true, y_pred, pos_label=1)

        precision = precision[:-1]
        recall = recall[:-1]
        indices = _downsample_thresholds_indices(thresholds.size, max_n_thresholds)
        precision, recall, thresholds = (
            precision[indices],
            recall[indices],
            thresholds[indices],
        )

        curve_data = {
            **metadata,
            "threshold": thresholds,
            "precision": precision,
            "recall": recall,
        }
        n = thresholds.size
        for col in metadata:
            curve_data[col] = Series([curve_data[col]], dtype="category").repeat(n)

        average_precision_df = DataFrame(
            {
                **metadata,
                "average_precision": [average_precision],
            }
        ).astype(dict.fromkeys(metadata, "category"))

        return DataFrame(curve_data), average_precision_df

    def frame(
        self,
        *,
        with_average_precision: bool = False,
        label: PositiveLabel = _DEFAULT,
    ) -> DataFrame:
        """Get the data used to create the precision-recall curve plot.

        Parameters
        ----------
        with_average_precision : bool, default=False
            Whether to include the average precision column in the returned DataFrame.

        label : int, float, bool, str or None, default=report pos_label
            The class whose curve to select. We always compute one curve per class,
            in a one-vs-rest fashion in multiclass classification and by alternating
            the positive class in binary classification. This lets you display only
            the curve of the desired class. Use `None` to show them all.

        Returns
        -------
        DataFrame
            A DataFrame containing the precision-recall curve data with columns
            depending on the report type:

            - `estimator`: Name of the estimator (when comparing estimators)
            - `split`: Cross-validation split ID (when doing cross-validation)
            - `label`: Class label (when plotting one-vs-rest curves)
            - `threshold`: Decision threshold
            - `precision`: Precision score at threshold
            - `recall`: Recall score at threshold
            - `data_source`: Data source used (when `data_source="both"`)
            - `average_precision`: average precision
              (when `with_average_precision=True`)

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> clf = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(clf, X, y, splitter=0.2)
        >>> display = report.metrics.precision_recall()
        >>> df = display.frame()
        """
        label = _check_label(self.labels, label, self.report_pos_label)

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
            indexing_columns = ["estimator"]
        else:  # self.report_type == "comparison-cross-validation"
            indexing_columns = ["estimator", "split"]

        if self.data_source == "both":
            indexing_columns += ["data_source"]

        if label is not None:
            df = df.query("label == @label").reset_index(drop=True)
            columns = indexing_columns + statistical_columns
        else:
            columns = indexing_columns + ["label"] + statistical_columns

        return df[columns]

    # ignore the type signature because we override kwargs by specifying the name of
    # the parameters for the user.
    def set_style(  # type: ignore[override]
        self,
        *,
        policy: Literal["override", "update"] = "update",
        relplot_kwargs: dict[str, Any] | None = None,
    ):
        """Set the style parameters for the display.

        Parameters
        ----------
        policy : {"override", "update"}, default="update"
            Policy to use when setting the style parameters.
            If "override", existing settings are set to the provided values.
            If "update", existing settings are not changed; only settings that were
            previously unset are changed.

        relplot_kwargs : dict, default=None
            Keyword arguments to be passed to :func:`seaborn.relplot` for rendering
            the precision-recall curve(s). Common options include `palette`,
            `alpha`, `linewidth`, etc.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a style parameter is unknown.
        """
        return super().set_style(
            policy=policy,
            relplot_kwargs=relplot_kwargs or {},
        )
