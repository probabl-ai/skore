from collections.abc import Sequence
from typing import Any, Literal, Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import LabelBinarizer

from skore.sklearn._plot.style import StyleDisplayMixin
from skore.sklearn._plot.utils import (
    LINESTYLE,
    HelpDisplayMixin,
    _ClassifierCurveDisplayMixin,
    _despine_matplotlib_axis,
    _validate_style_kwargs,
    sample_mpl_colormap,
)
from skore.sklearn.types import MLTask, PositiveLabel, ReportType, YPlotData


def _set_axis_labels(ax: Axes, info_pos_label: Union[str, None]) -> None:
    """Add axis labels."""
    xlabel = "Recall"
    ylabel = "Precision"
    if info_pos_label:
        xlabel += info_pos_label
        ylabel += info_pos_label

    ax.set(
        xlabel=xlabel,
        xlim=(-0.01, 1.01),
        ylabel=ylabel,
        ylim=(-0.01, 1.01),
        aspect="equal",
    )


class PrecisionRecallCurveDisplay(
    StyleDisplayMixin, HelpDisplayMixin, _ClassifierCurveDisplayMixin
):
    """Precision Recall visualization.

    An instance of this class is should created by
    `EstimatorReport.metrics.precision_recall()`. You should not create an
    instance of this class directly.

    Parameters
    ----------
    precision_recall : DataFrame
        The precision-recall curve data to display. The columns are
        - "estimator_name"
        - "split_index" (may be null)
        - "label"
        - "threshold"
        - "precision"
        - "recall".

    average_precision : DataFrame
        The average precision data to display. The columns are
        - "estimator_name"
        - "split_index" (may be null)
        - "label"
        - "average_precision".

    estimator_names : list of str
        Name of the estimators.

    pos_label : int, float, bool, str or None
        The class considered as the positive class. If None, the class will not
        be shown in the legend.

    data_source : {"train", "test", "X_y"}
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
    >>> display.plot(pr_curve_kwargs={"color": "tab:red"})
    """

    _default_pr_curve_kwargs: Union[dict[str, Any], None] = None

    def __init__(
        self,
        *,
        precision_recall: DataFrame,
        average_precision: DataFrame,
        pos_label: Optional[PositiveLabel],
        data_source: Literal["train", "test", "X_y"],
        ml_task: MLTask,
        report_type: ReportType,
    ) -> None:
        self.precision_recall = precision_recall
        self.average_precision = average_precision
        self.pos_label = pos_label
        self.data_source = data_source
        self.ml_task = ml_task
        self.report_type = report_type

    def _plot_single_estimator(
        self,
        *,
        estimator_name: str,
        pr_curve_kwargs: list[dict[str, Any]],
    ) -> tuple[Axes, list[Line2D], Union[str, None]]:
        """Plot precision-recall curve for a single estimator.

        Parameters
        ----------
        estimator_name : str
            The name of the estimator.

        pr_curve_kwargs : list of dict
            Additional keyword arguments to pass to matplotlib's plot function. In
            binary case, we should have a single dict. In multiclass case, we should
            have a list of dicts, one per class.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes containing the plot.

        lines : list of matplotlib.lines.Line2D
            The plotted lines.

        info_pos_label : str or None
            String containing the positive label info for binary classification.
            None for multiclass.
        """
        lines: list[Line2D] = []
        line_kwargs: dict[str, Any] = {"drawstyle": "steps-post"}

        if self.ml_task == "binary-classification":
            precision_recall = self.precision_recall.query(
                f"label == {self.pos_label!r}"
            )
            average_precision = self.average_precision["average_precision"].item()

            line_kwargs_validated = _validate_style_kwargs(
                line_kwargs, pr_curve_kwargs[0]
            )
            if self.data_source in ("train", "test"):
                line_kwargs_validated["label"] = (
                    f"{self.data_source.title()} set (AP = {average_precision:0.2f})"
                )
            else:  # data_source in (None, "X_y")
                line_kwargs_validated["label"] = f"AP = {average_precision:0.2f}"

            (line,) = self.ax_.plot(
                precision_recall["recall"],
                precision_recall["precision"],
                **line_kwargs_validated,
            )
            lines.append(line)

            info_pos_label = (
                f"\n(Positive label: {self.pos_label})"
                if self.pos_label is not None
                else ""
            )

        else:  # multiclass-classification
            labels = self.precision_recall["label"].cat.categories
            class_colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"),
                10 if len(labels) < 10 else len(labels),
            )

            for class_idx, class_label in enumerate(labels):
                query = f"label == {class_label!r}"
                precision_recall = self.precision_recall.query(query)
                average_precision = self.average_precision.query(query)[
                    "average_precision"
                ].item()
                pr_curve_kwargs_class = pr_curve_kwargs[class_idx]

                line_kwargs["color"] = class_colors[class_idx]
                line_kwargs_validated = _validate_style_kwargs(
                    line_kwargs, pr_curve_kwargs_class
                )
                if self.data_source in ("train", "test"):
                    line_kwargs_validated["label"] = (
                        f"{str(class_label).title()} - {self.data_source} "
                        f"set (AP = {average_precision:0.2f})"
                    )
                else:  # data_source in (None, "X_y")
                    line_kwargs_validated["label"] = (
                        f"{str(class_label).title()} - AP = {average_precision:0.2f}"
                    )

                (line,) = self.ax_.plot(
                    precision_recall["recall"],
                    precision_recall["precision"],
                    **line_kwargs_validated,
                )
                lines.append(line)

            info_pos_label = None  # irrelevant for multiclass

        self.ax_.legend(bbox_to_anchor=(1.02, 1), title=estimator_name)

        return self.ax_, lines, info_pos_label

    def _plot_cross_validated_estimator(
        self,
        *,
        estimator_name: str,
        pr_curve_kwargs: list[dict[str, Any]],
    ) -> tuple[Axes, list[Line2D], Union[str, None]]:
        """Plot precision-recall curve for a cross-validated estimator.

        Parameters
        ----------
        estimator_name : str
            The name of the estimator.

        pr_curve_kwargs : list of dict
            List of dictionaries containing keyword arguments to customize the
            precision-recall curves. The length of the list should match the number of
            curves to plot.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the precision-recall curves plotted.

        lines : list of matplotlib.lines.Line2D
            The plotted precision-recall curve lines.

        info_pos_label : str or None
            String containing positive label information for binary classification,
            None for multiclass.
        """
        lines: list[Line2D] = []
        line_kwargs: dict[str, Any] = {"drawstyle": "steps-post"}

        if self.ml_task == "binary-classification":
            for split_idx in self.precision_recall["split_index"].cat.categories:
                query = f"label == {self.pos_label!r} & split_index == {split_idx}"
                precision_recall = self.precision_recall.query(query)
                average_precision = self.average_precision.query(query)[
                    "average_precision"
                ].item()

                line_kwargs_validated = _validate_style_kwargs(
                    line_kwargs, pr_curve_kwargs[split_idx]
                )
                line_kwargs_validated["label"] = (
                    f"Estimator of fold #{split_idx + 1} "
                    f"(AP = {average_precision:0.2f})"
                )

                (line,) = self.ax_.plot(
                    precision_recall["recall"],
                    precision_recall["precision"],
                    **line_kwargs_validated,
                )
                lines.append(line)

            info_pos_label = (
                f"\n(Positive label: {self.pos_label})"
                if self.pos_label is not None
                else ""
            )
        else:  # multiclass-classification
            info_pos_label = None  # irrelevant for multiclass
            labels = self.precision_recall["label"].cat.categories
            class_colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"),
                10 if len(labels) < 10 else len(labels),
            )

            for class_idx, class_label in enumerate(labels):
                # precision_class = self.precision[class_]
                # recall_class = self.recall[class_]
                # average_precision_class = self.average_precision[class_]
                pr_curve_kwargs_class = pr_curve_kwargs[class_idx]

                for split_idx in self.precision_recall["split_index"].cat.categories:
                    query = f"label == {class_label!r} & split_index == {split_idx}"
                    precision_recall = self.precision_recall.query(query)
                    average_precision = self.average_precision.query(query)[
                        "average_precision"
                    ]
                    average_precision_mean = np.mean(average_precision)
                    average_precision_std = np.std(average_precision)

                    line_kwargs["color"] = class_colors[class_idx]
                    line_kwargs["alpha"] = 0.3
                    line_kwargs_validated = _validate_style_kwargs(
                        line_kwargs, pr_curve_kwargs_class
                    )
                    if split_idx == 0:
                        line_kwargs_validated["label"] = (
                            f"{str(class_label).title()} "
                            f"(AP = {average_precision_mean:0.2f} +/- "
                            f"{average_precision_std:0.2f})"
                        )
                    else:
                        line_kwargs_validated["label"] = None

                    (line,) = self.ax_.plot(
                        precision_recall["recall"],
                        precision_recall["precision"],
                        **line_kwargs_validated,
                    )
                    lines.append(line)

        if self.data_source in ("train", "test"):
            title = f"{estimator_name} on $\\bf{{{self.data_source}}}$ set"
        else:
            title = f"{estimator_name} on $\\bf{{external}}$ set"
        self.ax_.legend(bbox_to_anchor=(1.02, 1), title=title)

        return self.ax_, lines, info_pos_label

    def _plot_comparison_estimator(
        self,
        *,
        estimator_names: list[str],
        pr_curve_kwargs: list[dict[str, Any]],
    ) -> tuple[Axes, list[Line2D], Union[str, None]]:
        """Plot precision-recall curve of several estimators.

        Parameters
        ----------
        estimator_names : list of str
            The names of the estimators.

        pr_curve_kwargs : list of dict
            List of dictionaries containing keyword arguments to customize the
            precision-recall curves. The length of the list should match the number of
            curves to plot.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the precision-recall curves plotted.

        lines : list of matplotlib.lines.Line2D
            The plotted precision-recall curve lines.

        info_pos_label : str or None
            String containing positive label information for binary classification,
            None for multiclass.
        """
        lines: list[Line2D] = []
        line_kwargs: dict[str, Any] = {"drawstyle": "steps-post"}

        if self.ml_task == "binary-classification":
            for est_idx, est_name in enumerate(estimator_names):
                query = f"label == {self.pos_label!r} & estimator_name == '{est_name}'"
                precision_recall = self.precision_recall.query(query)
                average_precision = self.average_precision.query(query)[
                    "average_precision"
                ].item()

                line_kwargs_validated = _validate_style_kwargs(
                    line_kwargs, pr_curve_kwargs[est_idx]
                )
                line_kwargs_validated["label"] = (
                    f"{est_name} (AP = {average_precision:0.2f})"
                )
                (line,) = self.ax_.plot(
                    precision_recall["recall"],
                    precision_recall["precision"],
                    **line_kwargs_validated,
                )
                lines.append(line)

            info_pos_label = (
                f"\n(Positive label: {self.pos_label})"
                if self.pos_label is not None
                else ""
            )
        else:  # multiclass-classification
            info_pos_label = None  # irrelevant for multiclass
            labels = self.precision_recall["label"].cat.categories
            class_colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"),
                10 if len(labels) < 10 else len(labels),
            )

            for est_idx, est_name in enumerate(estimator_names):
                est_color = class_colors[est_idx]

                for class_idx, class_label in enumerate(labels):
                    query = f"label == {class_label!r} & estimator_name == '{est_name}'"
                    precision_recall = self.precision_recall.query(query)
                    average_precision = self.average_precision.query(query)[
                        "average_precision"
                    ].item()

                    class_linestyle = LINESTYLE[(class_idx % len(LINESTYLE))][1]
                    line_kwargs["color"] = est_color
                    line_kwargs["alpha"] = 0.6
                    line_kwargs["linestyle"] = class_linestyle
                    line_kwargs_validated = _validate_style_kwargs(
                        line_kwargs, pr_curve_kwargs[est_idx]
                    )
                    line_kwargs_validated["label"] = (
                        f"{est_name} - {str(class_label).title()} "
                        f"(AP = {average_precision:0.2f})"
                    )

                    (line,) = self.ax_.plot(
                        precision_recall["recall"],
                        precision_recall["precision"],
                        **line_kwargs_validated,
                    )
                    lines.append(line)

        self.ax_.legend(
            bbox_to_anchor=(1.02, 1),
            title=f"{self.ml_task.title()} on $\\bf{{{self.data_source}}}$ set",
        )

        return self.ax_, lines, info_pos_label

    def _plot_comparison_cross_validation(
        self,
        *,
        estimator_names: list[str],
        pr_curve_kwargs: list[dict[str, Any]],
    ) -> tuple[Axes, list[Line2D], Union[str, None]]:
        """Plot precision-recall curve of several estimators.

        Parameters
        ----------
        estimator_names : list of str
            The names of the estimators.

        pr_curve_kwargs : list of dict
            List of dictionaries containing keyword arguments to customize the
            precision-recall curves. The length of the list should match the number of
            curves to plot.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the precision-recall curves plotted.

        lines : list of matplotlib.lines.Line2D
            The plotted precision-recall curve lines.

        info_pos_label : str or None
            String containing positive label information for binary classification,
            None for multiclass.
        """
        lines: list[Line2D] = []
        line_kwargs: dict[str, Any] = {"drawstyle": "steps-post"}

        if self.ml_task == "binary-classification":
            labels = self.precision_recall["label"].cat.categories
            colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"),
                10 if len(estimator_names) < 10 else len(estimator_names),
            )
            curve_idx = 0
            for report_idx, estimator_name in enumerate(estimator_names):
                query = (
                    f"label == {self.pos_label!r} "
                    f"& estimator_name == '{estimator_name}'"
                )
                average_precision = self.average_precision.query(query)[
                    "average_precision"
                ]

                precision_recall = self.precision_recall.query(query)

                for split_idx, segment in precision_recall.groupby(
                    "split_index", observed=True
                ):
                    if split_idx == 0:
                        label_kwargs = {
                            "label": (
                                f"{estimator_name} "
                                f"(AUC = {average_precision.mean():0.2f} "
                                f"+/- {average_precision.std():0.2f})"
                            )
                        }
                    else:
                        label_kwargs = {}

                    line_kwargs["color"] = colors[report_idx]
                    line_kwargs["alpha"] = 0.6
                    line_kwargs_validated = _validate_style_kwargs(
                        line_kwargs, pr_curve_kwargs[curve_idx]
                    )

                    (line,) = self.ax_.plot(
                        segment["recall"],
                        segment["precision"],
                        **(line_kwargs_validated | label_kwargs),
                    )
                    lines.append(line)

                    curve_idx += 1

            info_pos_label = (
                f"\n(Positive label: {self.pos_label})"
                if self.pos_label is not None
                else ""
            )

            self.ax_.legend(
                bbox_to_anchor=(1.02, 1),
                title=f"{self.ml_task.title()} on $\\bf{{{self.data_source}}}$ set",
            )

        else:  # multiclass-classification
            info_pos_label = None  # irrelevant for multiclass
            labels = self.precision_recall["label"].cat.categories
            colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"),
                10 if len(estimator_names) < 10 else len(estimator_names),
            )
            idx = 0

            for est_idx, estimator_name in enumerate(estimator_names):
                est_color = colors[est_idx]

                for label_idx, label in enumerate(labels):
                    query = f"label == {label!r} & estimator_name == '{estimator_name}'"
                    average_precision = self.average_precision.query(query)[
                        "average_precision"
                    ]

                    precision_recall = self.precision_recall.query(query)

                    for split_idx, segment in precision_recall.groupby(
                        "split_index", observed=True
                    ):
                        if split_idx == 0:
                            label_kwargs = {
                                "label": (
                                    f"{estimator_name} "
                                    f"(AUC = {average_precision.mean():0.2f} "
                                    f"+/- {average_precision.std():0.2f})"
                                )
                            }
                        else:
                            label_kwargs = {}

                        line_kwargs["color"] = est_color
                        line_kwargs["alpha"] = 0.6
                        line_kwargs_validated = _validate_style_kwargs(
                            line_kwargs, pr_curve_kwargs[idx]
                        )

                        (line,) = self.ax_[label_idx].plot(
                            segment["recall"],
                            segment["precision"],
                            **(line_kwargs_validated | label_kwargs),
                        )
                        lines.append(line)

                        idx = idx + 1

                    info_pos_label = f"\n(Positive label: {label})"
                    _set_axis_labels(self.ax_[label_idx], info_pos_label)

            for ax in self.ax_:
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.2),
                    title=(
                        f"{self.ml_task.title()} on $\\bf{{{self.data_source}}}$ set"
                    ),
                )

        return self.ax_, lines, info_pos_label

    @StyleDisplayMixin.style_plot
    def plot(
        self,
        *,
        estimator_name: Optional[str] = None,
        pr_curve_kwargs: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None,
        despine: bool = True,
    ) -> None:
        """Plot visualization.

        Extra keyword arguments will be passed to matplotlib's `plot`.

        Parameters
        ----------
        estimator_name : str, default=None
            Name of the estimator used to plot the precision-recall curve. If
            `None`, we use the inferred name from the estimator.

        pr_curve_kwargs : dict or list of dict, default=None
            Keyword arguments to be passed to matplotlib's `plot` for rendering
            the precision-recall curve(s).

        despine : bool, default=True
            Whether to remove the top and right spines from the plot.

        Notes
        -----
        The average precision (cf. :func:`~sklearn.metrics.average_precision_score`)
        in scikit-learn is computed without any interpolation. To be consistent
        with this metric, the precision-recall curve is plotted without any
        interpolation as well (step-wise style).

        You can change this style by passing the keyword argument
        `drawstyle="default"`. However, the curve will not be strictly
        consistent with the reported average precision.

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
        >>> display.plot(pr_curve_kwargs={"color": "tab:red"})
        """
        if (
            self.report_type == "comparison-cross-validation"
            and self.ml_task == "multiclass-classification"
        ):
            n_labels = len(self.average_precision["label"].cat.categories)
            self.figure_, self.ax_ = plt.subplots(
                ncols=n_labels, figsize=(6.4 * n_labels, 4.8)
            )
        else:
            self.figure_, self.ax_ = plt.subplots()

        if pr_curve_kwargs is None:
            pr_curve_kwargs = self._default_pr_curve_kwargs

        if self.ml_task == "binary-classification":
            n_curves = len(self.average_precision.query(f"label == {self.pos_label!r}"))
        else:
            n_curves = len(self.average_precision)

        pr_curve_kwargs = self._validate_curve_kwargs(
            curve_param_name="pr_curve_kwargs",
            curve_kwargs=pr_curve_kwargs,
            n_curves=n_curves,
            report_type=self.report_type,
        )

        if self.report_type == "estimator":
            self.ax_, self.lines_, info_pos_label = self._plot_single_estimator(
                estimator_name=(
                    self.precision_recall["estimator_name"].cat.categories.item()
                    if estimator_name is None
                    else estimator_name
                ),
                pr_curve_kwargs=pr_curve_kwargs,
            )
        elif self.report_type == "cross-validation":
            self.ax_, self.lines_, info_pos_label = (
                self._plot_cross_validated_estimator(
                    estimator_name=(
                        self.precision_recall["estimator_name"].cat.categories.item()
                        if estimator_name is None
                        else estimator_name
                    ),
                    pr_curve_kwargs=pr_curve_kwargs,
                )
            )
        elif self.report_type == "comparison-estimator":
            self.ax_, self.lines_, info_pos_label = self._plot_comparison_estimator(
                estimator_names=self.precision_recall["estimator_name"].cat.categories,
                pr_curve_kwargs=pr_curve_kwargs,
            )
        elif self.report_type == "comparison-cross-validation":
            self.ax_, self.lines_, info_pos_label = (
                self._plot_comparison_cross_validation(
                    estimator_names=self.precision_recall[
                        "estimator_name"
                    ].cat.categories,
                    pr_curve_kwargs=pr_curve_kwargs,
                )
            )
        else:
            raise ValueError(
                "`report_type` should be one of 'estimator', 'cross-validation', "
                "'comparison-cross-validation' or 'comparison-estimator'. "
                f"Got '{self.report_type}' instead."
            )

        if (
            self.report_type == "comparison-cross-validation"
            and self.ml_task == "multiclass-classification"
        ):
            for ax in self.ax_:
                if despine:
                    _despine_matplotlib_axis(ax)
        else:
            _set_axis_labels(self.ax_, info_pos_label)

            if despine:
                _despine_matplotlib_axis(self.ax_)

    @classmethod
    def _compute_data_for_display(
        cls,
        y_true: Sequence[YPlotData],
        y_pred: Sequence[YPlotData],
        *,
        report_type: ReportType,
        estimators: Sequence[BaseEstimator],
        estimator_names: list[str],
        ml_task: MLTask,
        data_source: Literal["train", "test", "X_y"],
        pos_label: Optional[PositiveLabel],
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

        estimator_names : list of str
            Name of the estimators used to plot the precision-recall curve.

        ml_task : {"binary-classification", "multiclass-classification"}
            The machine learning task.

        data_source : {"train", "test", "X_y"}
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
            for y_true_i, y_pred_i in zip(y_true, y_pred):
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
                    precision_i, recall_i, thresholds_i
                ):
                    precision_recall_records.append(
                        {
                            "estimator_name": y_true_i.estimator_name,
                            "split_index": y_true_i.split_index,
                            "label": pos_label_validated,
                            "threshold": threshold,
                            "precision": precision,
                            "recall": recall,
                        }
                    )
                average_precision_records.append(
                    {
                        "estimator_name": y_true_i.estimator_name,
                        "split_index": y_true_i.split_index,
                        "label": pos_label_validated,
                        "average_precision": average_precision_i,
                    }
                )
        else:  # multiclass-classification
            for y_true_i, y_pred_i, est in zip(y_true, y_pred, estimators):
                label_binarizer = LabelBinarizer().fit(est.classes_)
                y_true_onehot_i: NDArray = label_binarizer.transform(y_true_i.y)
                for class_idx, class_ in enumerate(est.classes_):
                    precision_class_i, recall_class_i, thresholds_class_i = (
                        precision_recall_curve(
                            y_true_onehot_i[:, class_idx],
                            y_pred_i.y[:, class_idx],
                            pos_label=None,
                            drop_intermediate=drop_intermediate,
                        )
                    )
                    average_precision_class_i = average_precision_score(
                        y_true_onehot_i[:, class_idx], y_pred_i.y[:, class_idx]
                    )

                    for precision, recall, threshold in zip(
                        precision_class_i, recall_class_i, thresholds_class_i
                    ):
                        precision_recall_records.append(
                            {
                                "estimator_name": y_true_i.estimator_name,
                                "split_index": y_true_i.split_index,
                                "label": class_,
                                "threshold": threshold,
                                "precision": precision,
                                "recall": recall,
                            }
                        )
                    average_precision_records.append(
                        {
                            "estimator_name": y_true_i.estimator_name,
                            "split_index": y_true_i.split_index,
                            "label": class_,
                            "average_precision": average_precision_class_i,
                        }
                    )

        dtypes = {
            "estimator_name": "category",
            "split_index": "category",
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
