from collections import defaultdict
from typing import Any, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import LabelBinarizer

from skore.sklearn._comparison.roc_curve_display import LINESTYLE
from skore.sklearn._plot.utils import (
    HelpDisplayMixin,
    _ClassifierCurveDisplayMixin,
    _despine_matplotlib_axis,
    sample_mpl_colormap,
)


class PrecisionRecallCurveDisplay(HelpDisplayMixin, _ClassifierCurveDisplayMixin):
    """Precision Recall visualization.

    An instance of this class is should created by
    `ComparisonReport.metrics.precision_recall()`.
    You should not create an instance of this class directly.

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

    Attributes
    ----------
    ax_ : matplotlib Axes
        Axes with precision recall curve, available after calling `plot`.

    figure_ : matplotlib Figure
        Figure containing the curve, available after calling `plot`.

    lines_ : list of matplotlib lines
        The lines of the precision recall curve, available after calling `plot`.
    """

    def __init__(
        self,
        *,
        precision: dict[Any, list[ArrayLike]],
        recall: dict[Any, list[ArrayLike]],
        average_precision: dict[Any, list[float]],
        estimator_names: list[str],
        ml_task: Literal["binary-classification", "multiclass-classification"],
        data_source: Literal["train", "test", "X_y"],
        pos_label: Any = None,
    ):
        self.precision = precision
        self.recall = recall
        self.average_precision = average_precision
        self.estimator_names = estimator_names
        self.ml_task = ml_task
        self.data_source = data_source
        self.pos_label = pos_label

    def plot(
        self,
        ax: Optional[Axes] = None,
        *,
        despine: bool = True,
    ):
        """Plot visualization.

        Parameters
        ----------
        ax : Matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        despine : bool, default=True
            Whether to remove the top and right spines from the plot.

        Notes
        -----
        The average precision (cf. :func:`~sklearn.metrics.average_precision_score`)
        in scikit-learn is computed without any interpolation. To be consistent
        with this metric, the precision-recall curve is plotted without any
        interpolation as well (step-wise style).
        """
        self.figure_, self.ax_ = (ax.figure, ax) if ax else plt.subplots()
        self.lines_ = []

        if self.ml_task == "binary-classification":
            for report_idx, report_name in enumerate(self.estimator_names):
                precision = self.precision[self.pos_label][report_idx]
                recall = self.recall[self.pos_label][report_idx]
                average_precision = self.average_precision[self.pos_label][report_idx]

                self.lines_ += self.ax_.plot(
                    recall,
                    precision,
                    drawstyle="steps-post",
                    alpha=0.6,
                    label=(
                        f"{report_name} #{report_idx + 1} "
                        f"(AP = {average_precision:0.2f})"
                    ),
                )

            info_pos_label = (
                f"\n(Positive label: {self.pos_label})"
                if self.pos_label is not None
                else ""
            )
        else:  # multiclass-classification
            info_pos_label = None  # irrelevant for multiclass
            colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"),
                10 if len(self.estimator_names) < 10 else len(self.estimator_names),
            )

            for report_idx, report_name in enumerate(self.estimator_names):
                report_color = colors[report_idx]

                for class_idx, class_ in enumerate(self.precision):
                    precision = self.precision[class_][report_idx]
                    recall = self.recall[class_][report_idx]
                    average_precision_class = self.average_precision[class_]
                    average_precision = average_precision_class[report_idx]
                    class_linestyle = LINESTYLE[(class_idx % len(LINESTYLE))][1]

                    self.lines_ += self.ax_.plot(
                        recall,
                        precision,
                        color=report_color,
                        linestyle=class_linestyle,
                        alpha=0.6,
                        label=(
                            f"{report_name} #{report_idx + 1} - class {class_} "
                            f"(AP = {np.mean(average_precision_class):0.2f})"
                        ),
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

        self.ax_.legend(
            loc="lower right",
            title=f"{self.ml_task.title()} on $\\bf{{{self.data_source}}}$ set",
        )

    @classmethod
    def _from_predictions(
        cls,
        y_true: list[ArrayLike],
        y_pred: list[NDArray],
        *,
        estimators: list[BaseEstimator],
        estimator_names: list[str],
        ml_task: Literal["binary-classification", "multiclass-classification"],
        data_source: Literal["train", "test", "X_y"],
        pos_label: Union[int, float, bool, str, None],
        drop_intermediate: bool = True,
    ):
        """Private factory to create a PrecisionRecallCurveDisplay from predictions.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels.

        y_pred : array-like of shape (n_samples,)
            Target scores, can either be probability estimates of the positive class,
            confidence values, or non-thresholded measure of decisions (as returned by
            “decision_function” on some classifiers).

        estimators : list of estimator instances
            The estimators from which `y_pred` is obtained.

        estimator_names : list[str]
            Name of the estimators used to plot the precision recall curve.

        ml_task : {"binary-classification", "multiclass-classification"}
            The machine learning task.

        data_source : {"train", "test", "X_y"}
            The data source used to compute the precision recall curve.

        pos_label : int, float, bool or str, default=None
            The class considered as the positive class when computing the
            precision and recall metrics.

        drop_intermediate : bool, default=False
            Whether to drop some suboptimal thresholds which would not appear
            on a plotted precision-recall curve. This is useful in order to
            create lighter precision-recall curves.

        Returns
        -------
        display : PrecisionRecallCurveDisplay
        """
        estimator_classes = [estimator.classes_ for estimator in estimators]
        precision, recall, average_precision = (defaultdict(list) for _ in range(3))
        pos_label_validated = cls._validate_from_predictions_params(
            y_true, y_pred, ml_task=ml_task, pos_label=pos_label
        )

        if ml_task == "binary-classification":
            for y_true_i, y_pred_i in zip(y_true, y_pred):
                precision_i, recall_i, _ = precision_recall_curve(
                    y_true_i,
                    y_pred_i,
                    pos_label=pos_label_validated,
                    drop_intermediate=drop_intermediate,
                )

                precision[pos_label_validated].append(precision_i)
                recall[pos_label_validated].append(recall_i)
                average_precision[pos_label_validated].append(
                    average_precision_score(
                        y_true_i,
                        y_pred_i,
                        pos_label=pos_label_validated,
                    )
                )
        elif ml_task == "multiclass-classification":
            for y_true_i, y_pred_i, estimator_classes_i in zip(
                y_true,
                y_pred,
                estimator_classes,
            ):
                label_binarizer = LabelBinarizer().fit(estimator_classes_i)
                y_true_onehot_i = label_binarizer.transform(y_true_i)

                for class_idx, class_ in enumerate(estimator_classes_i):
                    precision_class_i, recall_class_i, _ = precision_recall_curve(
                        y_true_onehot_i[:, class_idx],
                        y_pred_i[:, class_idx],
                        pos_label=None,
                        drop_intermediate=drop_intermediate,
                    )

                    precision[class_].append(precision_class_i)
                    recall[class_].append(recall_class_i)
                    average_precision[class_].append(
                        average_precision_score(
                            y_true_onehot_i[:, class_idx],
                            y_pred_i[:, class_idx],
                        )
                    )
        else:
            raise ValueError("Only binary or multiclass classification is allowed")

        return cls(
            precision=dict(precision),
            recall=dict(recall),
            average_precision=dict(average_precision),
            estimator_names=estimator_names,
            ml_task=ml_task,
            pos_label=pos_label_validated,
            data_source=data_source,
        )
