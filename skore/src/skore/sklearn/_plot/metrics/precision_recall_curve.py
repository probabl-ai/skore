from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal, Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike, NDArray
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
from skore.sklearn.types import MLTask, PositiveLabel, YPlotData


class PrecisionRecallCurveDisplay(
    StyleDisplayMixin, HelpDisplayMixin, _ClassifierCurveDisplayMixin
):
    """Precision Recall visualization.

    An instance of this class is should created by
    `EstimatorReport.metrics.precision_recall()`. You should not create an
    instance of this class directly.


    Parameters
    ----------
    precision : dict of list of ndarray
        Precision values. The structure is:

        - for binary classification:
            - the key is the positive label.
            - the value is a list of `ndarray`, each `ndarray` being the precision.
        - for multiclass classification:
            - the key is the class of interest in an OvR fashion.
            - the value is a list of `ndarray`, each `ndarray` being the precision.

    recall : dict of list of ndarray
        Recall values. The structure is:

        - for binary classification:
            - the key is the positive label.
            - the value is a list of `ndarray`, each `ndarray` being the recall.
        - for multiclass classification:
            - the key is the class of interest in an OvR fashion.
            - the value is a list of `ndarray`, each `ndarray` being the recall.

    average_precision : dict of list of float
        Average precision. The structure is:

        - for binary classification:
            - the key is the positive label.
            - the value is a list of `float`, each `float` being the average
              precision.
        - for multiclass classification:
            - the key is the class of interest in an OvR fashion.
            - the value is a list of `float`, each `float` being the average
              precision.

    estimator_names : list of str
        Name of the estimators.

    pos_label : int, float, bool, str or None
        The class considered as the positive class. If None, the class will not
        be shown in the legend.

    data_source : {"train", "test", "X_y"}
        The data source used to compute the precision recall curve.

    ml_task : {"binary-classification", "multiclass-classification"}
        The machine learning task.

    report_type : {"comparison-estimator", "cross-validation", "estimator"}
        The type of report.

    Attributes
    ----------
    ax_ : matplotlib Axes
        Axes with precision recall curve.

    figure_ : matplotlib Figure
        Figure containing the curve.

    lines_ : list of matplotlib Artist
        Precision recall curve.

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
        precision: dict[Any, list[ArrayLike]],
        recall: dict[Any, list[ArrayLike]],
        average_precision: dict[Any, list[float]],
        estimator_names: list[str],
        pos_label: Optional[PositiveLabel],
        data_source: Literal["train", "test", "X_y"],
        ml_task: MLTask,
        report_type: Literal["comparison-estimator", "cross-validation", "estimator"],
    ) -> None:
        self.precision = precision
        self.recall = recall
        self.average_precision = average_precision
        self.estimator_names = estimator_names
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
            pos_label = cast(PositiveLabel, self.pos_label)

            line_kwargs_validated = _validate_style_kwargs(
                line_kwargs, pr_curve_kwargs[0]
            )
            if self.data_source in ("train", "test"):
                line_kwargs_validated["label"] = (
                    f"{self.data_source.title()} set "
                    f"(AP = {self.average_precision[pos_label][0]:0.2f})"
                )
            else:  # data_source in (None, "X_y")
                line_kwargs_validated["label"] = (
                    f"AP = {self.average_precision[pos_label][0]:0.2f}"
                )

            (line,) = self.ax_.plot(
                self.recall[pos_label][0],
                self.precision[pos_label][0],
                **line_kwargs_validated,
            )
            lines.append(line)

            info_pos_label = (
                f"\n(Positive label: {pos_label})" if pos_label is not None else ""
            )

        else:  # multiclass-classification
            class_colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"),
                10 if len(self.precision) < 10 else len(self.precision),
            )

            for class_idx, class_label in enumerate(self.precision):
                recall_class = self.recall[class_label][0]
                precision_class = self.precision[class_label][0]
                average_precision_class = self.average_precision[class_label][0]
                pr_curve_kwargs_class = pr_curve_kwargs[class_idx]

                line_kwargs["color"] = class_colors[class_idx]
                line_kwargs_validated = _validate_style_kwargs(
                    line_kwargs, pr_curve_kwargs_class
                )
                if self.data_source in ("train", "test"):
                    line_kwargs_validated["label"] = (
                        f"{str(class_label).title()} - {self.data_source} "
                        f"set (AP = {average_precision_class:0.2f})"
                    )
                else:  # data_source in (None, "X_y")
                    line_kwargs_validated["label"] = (
                        f"{str(class_label).title()} - "
                        f"AP = {average_precision_class:0.2f}"
                    )

                (line,) = self.ax_.plot(
                    recall_class, precision_class, **line_kwargs_validated
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
            pos_label = cast(PositiveLabel, self.pos_label)
            for split_idx in range(len(self.precision[pos_label])):
                precision_split = self.precision[pos_label][split_idx]
                recall_split = self.recall[pos_label][split_idx]
                average_precision_split = self.average_precision[pos_label][split_idx]

                line_kwargs_validated = _validate_style_kwargs(
                    line_kwargs, pr_curve_kwargs[split_idx]
                )
                line_kwargs_validated["label"] = (
                    f"Estimator of fold #{split_idx + 1} "
                    f"(AP = {average_precision_split:0.2f})"
                )

                (line,) = self.ax_.plot(
                    recall_split, precision_split, **line_kwargs_validated
                )
                lines.append(line)

            info_pos_label = (
                f"\n(Positive label: {pos_label})" if pos_label is not None else ""
            )
        else:  # multiclass-classification
            info_pos_label = None  # irrelevant for multiclass
            class_colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"),
                10 if len(self.precision) < 10 else len(self.precision),
            )

            for class_idx, class_ in enumerate(self.precision):
                precision_class = self.precision[class_]
                recall_class = self.recall[class_]
                average_precision_class = self.average_precision[class_]
                pr_curve_kwargs_class = pr_curve_kwargs[class_idx]

                for split_idx in range(len(precision_class)):
                    precision_split = precision_class[split_idx]
                    recall_split = recall_class[split_idx]
                    average_precision_mean = np.mean(average_precision_class)
                    average_precision_std = np.std(average_precision_class)

                    line_kwargs["color"] = class_colors[class_idx]
                    line_kwargs["alpha"] = 0.3
                    line_kwargs_validated = _validate_style_kwargs(
                        line_kwargs, pr_curve_kwargs_class
                    )
                    if split_idx == 0:
                        line_kwargs_validated["label"] = (
                            f"{str(class_).title()} "
                            f"(AP = {average_precision_mean:0.2f} +/- "
                            f"{average_precision_std:0.2f})"
                        )
                    else:
                        line_kwargs_validated["label"] = None

                    (line,) = self.ax_.plot(
                        recall_split, precision_split, **line_kwargs_validated
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
            pos_label = cast(PositiveLabel, self.pos_label)
            for est_idx, est_name in enumerate(estimator_names):
                precision_est = self.precision[pos_label][est_idx]
                recall_est = self.recall[pos_label][est_idx]
                average_precision_est = self.average_precision[pos_label][est_idx]

                line_kwargs_validated = _validate_style_kwargs(
                    line_kwargs, pr_curve_kwargs[est_idx]
                )
                line_kwargs_validated["label"] = (
                    f"{est_name} (AP = {average_precision_est:0.2f})"
                )
                (line,) = self.ax_.plot(
                    recall_est, precision_est, **line_kwargs_validated
                )
                lines.append(line)

            info_pos_label = (
                f"\n(Positive label: {pos_label})" if pos_label is not None else ""
            )
        else:  # multiclass-classification
            info_pos_label = None  # irrelevant for multiclass
            class_colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"),
                10 if len(self.precision) < 10 else len(self.precision),
            )

            for est_idx, est_name in enumerate(estimator_names):
                est_color = class_colors[est_idx]

                for class_idx, class_ in enumerate(self.precision):
                    precision_est_class = self.precision[class_][est_idx]
                    recall_est_class = self.recall[class_][est_idx]
                    average_precision_mean = self.average_precision[class_][est_idx]
                    class_linestyle = LINESTYLE[(class_idx % len(LINESTYLE))][1]

                    line_kwargs["color"] = est_color
                    line_kwargs["alpha"] = 0.6
                    line_kwargs["linestyle"] = class_linestyle

                    line_kwargs_validated = _validate_style_kwargs(
                        line_kwargs, pr_curve_kwargs[est_idx]
                    )
                    line_kwargs_validated["label"] = (
                        f"{est_name} - {str(class_).title()} "
                        f"(AP = {average_precision_mean:0.2f})"
                    )

                    (line,) = self.ax_.plot(
                        recall_est_class, precision_est_class, **line_kwargs_validated
                    )
                    lines.append(line)

        self.ax_.legend(
            bbox_to_anchor=(1.02, 1),
            title=f"{self.ml_task.title()} on $\\bf{{{self.data_source}}}$ set",
        )

        return self.ax_, lines, info_pos_label

    @StyleDisplayMixin.style_plot
    def plot(
        self,
        ax: Optional[Axes] = None,
        *,
        estimator_name: Optional[str] = None,
        pr_curve_kwargs: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None,
        despine: bool = True,
    ) -> None:
        """Plot visualization.

        Extra keyword arguments will be passed to matplotlib's `plot`.

        Parameters
        ----------
        ax : Matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

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
        self.figure_, self.ax_ = (ax.figure, ax) if ax is not None else plt.subplots()

        if pr_curve_kwargs is None:
            pr_curve_kwargs = self._default_pr_curve_kwargs

        pr_curve_kwargs = self._validate_curve_kwargs(
            curve_param_name="pr_curve_kwargs",
            curve_kwargs=pr_curve_kwargs,
            metric=self.average_precision,
            report_type=self.report_type,
        )

        if self.report_type == "estimator":
            self.ax_, self.lines_, info_pos_label = self._plot_single_estimator(
                estimator_name=(
                    self.estimator_names[0]
                    if estimator_name is None
                    else estimator_name
                ),
                pr_curve_kwargs=pr_curve_kwargs,
            )
        elif self.report_type == "cross-validation":
            self.ax_, self.lines_, info_pos_label = (
                self._plot_cross_validated_estimator(
                    estimator_name=(
                        self.estimator_names[0]
                        if estimator_name is None
                        else estimator_name
                    ),
                    pr_curve_kwargs=pr_curve_kwargs,
                )
            )
        elif self.report_type == "comparison-estimator":
            self.ax_, self.lines_, info_pos_label = self._plot_comparison_estimator(
                estimator_names=self.estimator_names,
                pr_curve_kwargs=pr_curve_kwargs,
            )
        else:
            raise ValueError(
                f"`report_type` should be one of 'estimator', 'cross-validation', "
                f"or 'comparison-estimator'. Got '{self.report_type}' instead."
            )

        xlabel = "Recall"
        ylabel = "Precision"
        if info_pos_label:
            xlabel += info_pos_label
            ylabel += info_pos_label

        self.ax_.set(
            xlabel=xlabel,
            xlim=(-0.01, 1.01),
            ylabel=ylabel,
            ylim=(-0.01, 1.01),
            aspect="equal",
        )

        if despine:
            _despine_matplotlib_axis(self.ax_)

    @classmethod
    def _compute_data_for_display(
        cls,
        y_true: Sequence[YPlotData],
        y_pred: Sequence[YPlotData],
        *,
        report_type: Literal["comparison-estimator", "cross-validation", "estimator"],
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

        report_type : {"comparison-estimator", "cross-validation", "estimator"}
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

        precision: dict[PositiveLabel, list[ArrayLike]] = defaultdict(list)
        recall: dict[PositiveLabel, list[ArrayLike]] = defaultdict(list)
        average_precision: dict[PositiveLabel, list[float]] = defaultdict(list)

        if ml_task == "binary-classification":
            for y_true_i, y_pred_i in zip(y_true, y_pred):
                pos_label_validated = cast(PositiveLabel, pos_label_validated)
                precision_i, recall_i, _ = precision_recall_curve(
                    y_true_i.y,
                    y_pred_i.y,
                    pos_label=pos_label_validated,
                    drop_intermediate=drop_intermediate,
                )
                average_precision_i = average_precision_score(
                    y_true_i.y, y_pred_i.y, pos_label=pos_label_validated
                )

                precision[pos_label_validated].append(precision_i)
                recall[pos_label_validated].append(recall_i)
                average_precision[pos_label_validated].append(average_precision_i)
        else:  # multiclass-classification
            for y_true_i, y_pred_i, est in zip(y_true, y_pred, estimators):
                label_binarizer = LabelBinarizer().fit(est.classes_)
                y_true_onehot_i: NDArray = label_binarizer.transform(y_true_i.y)
                for class_idx, class_ in enumerate(est.classes_):
                    precision_class_i, recall_class_i, _ = precision_recall_curve(
                        y_true_onehot_i[:, class_idx],
                        y_pred_i.y[:, class_idx],
                        pos_label=None,
                        drop_intermediate=drop_intermediate,
                    )
                    average_precision_class_i = average_precision_score(
                        y_true_onehot_i[:, class_idx], y_pred_i.y[:, class_idx]
                    )

                    precision[class_].append(precision_class_i)
                    recall[class_].append(recall_class_i)
                    average_precision[class_].append(average_precision_class_i)

        return cls(
            precision=precision,
            recall=recall,
            average_precision=average_precision,
            estimator_names=estimator_names,
            pos_label=pos_label_validated,
            data_source=data_source,
            ml_task=ml_task,
            report_type=report_type,
        )
