from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelBinarizer

from skore.sklearn._plot.utils import (
    HelpDisplayMixin,
    _ClassifierCurveDisplayMixin,
    _despine_matplotlib_axis,
    sample_mpl_colormap,
)

LINESTYLE = [
    ("solid", "solid"),
    ("dotted", "dotted"),
    ("dashed", "dashed"),
    ("dashdot", "dashdot"),
    ("loosely dotted", (0, (1, 10))),
    ("dotted", (0, (1, 5))),
    ("densely dotted", (0, (1, 1))),
    ("long dash with offset", (5, (10, 3))),
    ("loosely dashed", (0, (5, 10))),
    ("dashed", (0, (5, 5))),
    ("densely dashed", (0, (5, 1))),
    ("loosely dashdotted", (0, (3, 10, 1, 10))),
    ("dashdotted", (0, (3, 5, 1, 5))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
]


class RocCurveDisplay(HelpDisplayMixin, _ClassifierCurveDisplayMixin):
    """ROC Curve visualization for comparison report.

    An instance of this class is should created by `ComparisonReport.metrics.roc()`.
    You should not create an instance of this class directly.

    Parameters
    ----------
    fpr : dict of list of ndarray
        False positive rate. The structure is:

        - for binary classification:
            - the key is the positive label.
            - the value is a list of `ndarray`, each `ndarray` being the false
              positive rate.
        - for multiclass classification:
            - the key is the class of interest in an OvR fashion.
            - the value is a list of `ndarray`, each `ndarray` being the false
              positive rate.

    tpr : dict of list of ndarray
        True positive rate. The structure is:

        - for binary classification:
            - the key is the positive label
            - the value is a list of `ndarray`, each `ndarray` being the true
              positive rate.
        - for multiclass classification:
            - the key is the class of interest in an OvR fashion.
            - the value is a list of `ndarray`, each `ndarray` being the true
              positive rate.

    roc_auc : dict of list of float
        Area under the ROC curve. The structure is:

        - for binary classification:
            - the key is the positive label
            - the value is a list of `float`, each `float` being the area under
              the ROC curve.
        - for multiclass classification:
            - the key is the class of interest in an OvR fashion.
            - the value is a list of `float`, each `float` being the area under
              the ROC curve.

    estimator_names : str
        Name of the estimators.

    ml_task : str
        Type of ML task.

    pos_label : int, float, bool or str, default=None
        The class considered as positive. Only meaningful for binary classification.

    data_source : {"train", "test", "X_y"}
        The data source used to compute the ROC curve.

    Attributes
    ----------
    ax_ : matplotlib axes
        The axes on which the ROC curve is plotted, available after calling `plot`.

    figure_ : matplotlib figure
        The figure on which the ROC curve is plotted, available after calling `plot`.

    lines_ : list of matplotlib lines
        The lines of the ROC curve, available after calling `plot`.
    """

    def __init__(
        self,
        *,
        fpr,
        tpr,
        roc_auc,
        estimator_names,
        ml_task,
        data_source,
        pos_label=None,
    ):
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc
        self.estimator_names = estimator_names
        self.ml_task = ml_task
        self.data_source = data_source
        self.pos_label = pos_label

    def plot(
        self,
        ax=None,
        *,
        plot_chance_level=True,
        despine=True,
    ):
        """Plot visualization.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is created.
        plot_chance_level : bool, default=True
            Whether to plot the chance level.
        despine : bool, default=True
            Whether to remove the top and right spines from the plot.
        """
        self.figure_, self.ax_ = (ax.figure, ax) if ax else plt.subplots()
        self.lines_ = []

        if self.ml_task == "binary-classification":
            for report_idx, report_name in enumerate(self.estimator_names):
                fpr = self.fpr[self.pos_label][report_idx]
                tpr = self.tpr[self.pos_label][report_idx]
                roc_auc = self.roc_auc[self.pos_label][report_idx]

                self.lines_ += self.ax_.plot(
                    fpr,
                    tpr,
                    alpha=0.6,
                    label=f"{report_name} #{report_idx + 1} (AUC = {roc_auc:0.2f})",
                )

            info_pos_label = (
                f"\n(Positive label: {self.pos_label})"
                if self.pos_label is not None
                else ""
            )
        else:  # multiclass-classification
            info_pos_label = None  # irrelevant for multiclass
            colors = sample_mpl_colormap(
                plt.cm.tab10,
                10 if len(self.estimator_names) < 10 else len(self.estimator_names),
            )

            for report_idx, report_name in enumerate(self.estimator_names):
                report_color = colors[report_idx]

                for class_idx, class_ in enumerate(self.fpr):
                    fpr = self.fpr[class_][report_idx]
                    tpr = self.tpr[class_][report_idx]
                    roc_auc_mean = np.mean(self.roc_auc[class_])
                    class_linestyle = LINESTYLE[(class_idx % len(LINESTYLE))][1]

                    self.lines_ += self.ax_.plot(
                        fpr,
                        tpr,
                        alpha=0.6,
                        linestyle=class_linestyle,
                        color=report_color,
                        label=(
                            f"{report_name} #{report_idx + 1} - class {class_} "
                            f"(AUC = {roc_auc_mean:0.2f})"
                        ),
                    )

        xlabel = "False Positive Rate"
        ylabel = "True Positive Rate"
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

        if plot_chance_level:
            self.ax_.plot(
                (0, 1),
                (0, 1),
                label="Chance level (AUC = 0.5)",
                color="k",
                linestyle="--",
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
        y_true,
        y_pred,
        *,
        estimators,
        estimator_names,
        ml_task,
        data_source,
        pos_label=None,
        drop_intermediate=True,
    ):
        """Private factory to create a RocCurveDisplay from predictions.

        Parameters
        ----------
        y_true : list of array-like of shape (n_samples,)
            True binary labels in binary classification.

        y_pred : list of array-like of shape (n_samples,)
            Target scores, can either be probability estimates of the positive class,
            confidence values, or non-thresholded measure of decisions (as returned by
            “decision_function” on some classifiers).

        estimators : list of estimator instances
            The estimators from which `y_pred` is obtained.

        estimator_names : list[str]
            Name of the estimators used to plot the ROC curve.

        ml_task : {"binary-classification", "multiclass-classification"}
            The machine learning task.

        data_source : {"train", "test", "X_y"}
            The data source used to compute the ROC curve.

        pos_label : int, float, bool or str, default=None
            The class considered as the positive class when computing the
            precision and recall metrics.

        drop_intermediate : bool, default=True
            Whether to drop intermediate points with identical value.

        Returns
        -------
        display : RocCurveDisplay
        """
        estimator_classes = [estimator.classes_ for estimator in estimators]
        fpr, tpr, roc_auc = (defaultdict(list) for _ in range(3))
        pos_label_validated = cls._validate_from_predictions_params(
            y_true, y_pred, ml_task=ml_task, pos_label=pos_label
        )

        if ml_task == "binary-classification":
            for y_true_i, y_pred_i in zip(y_true, y_pred):
                fpr_i, tpr_i, _ = roc_curve(
                    y_true_i,
                    y_pred_i,
                    pos_label=pos_label,
                    drop_intermediate=drop_intermediate,
                )

                fpr[pos_label_validated].append(fpr_i)
                tpr[pos_label_validated].append(tpr_i)
                roc_auc[pos_label_validated].append(auc(fpr_i, tpr_i))
        elif ml_task == "multiclass-classification":
            for y_true_i, y_pred_i, estimator_classes_i in zip(
                y_true,
                y_pred,
                estimator_classes,
            ):
                label_binarizer = LabelBinarizer().fit(estimator_classes_i)
                y_true_onehot_i = label_binarizer.transform(y_true_i)

                for class_idx, class_ in enumerate(estimator_classes_i):
                    fpr_class_i, tpr_class_i, _ = roc_curve(
                        y_true_onehot_i[:, class_idx],
                        y_pred_i[:, class_idx],
                        pos_label=None,
                        drop_intermediate=drop_intermediate,
                    )

                    fpr[class_].append(fpr_class_i)
                    tpr[class_].append(tpr_class_i)
                    roc_auc[class_].append(auc(fpr_class_i, tpr_class_i))
        else:
            raise ValueError("Only binary or multiclass classification is allowed")

        return cls(
            fpr=dict(fpr),
            tpr=dict(tpr),
            roc_auc=dict(roc_auc),
            estimator_names=estimator_names,
            ml_task=ml_task,
            pos_label=pos_label_validated,
            data_source=data_source,
        )
