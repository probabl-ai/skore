from collections import defaultdict
from operator import attrgetter

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
    "solid",
    "dotted",
    "dashed",
    "dashdot",
    (0, (1, 10)),
    (0, (1, 5)),
    (0, (1, 1)),
    (5, (10, 3)),
    (0, (5, 10)),
    (0, (5, 5)),
    (0, (5, 1)),
    (0, (3, 10, 1, 10)),
    (0, (3, 5, 1, 5)),
    (0, (3, 1, 1, 1)),
    (0, (3, 5, 1, 5, 1, 5)),
    (0, (3, 10, 1, 10, 1, 10)),
    (0, (3, 1, 1, 1, 1, 1)),
]


class RocCurveDisplay(HelpDisplayMixin, _ClassifierCurveDisplayMixin):
    def __init__(
        self,
        *,
        fpr,
        tpr,
        roc_auc,
        estimator_names,
        ml_task,
        pos_label=None,
        data_source=None,
    ):
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc
        self.estimator_names = estimator_names
        self.ml_task = ml_task
        self.pos_label = pos_label
        self.data_source = data_source

    def plot(
        self,
        *,
        plot_chance_level=True,
        despine=True,
    ):
        _, ax = plt.subplots()

        if self.ml_task == "binary-classification":
            for report_idx, report_name in enumerate(self.estimator_names):
                fpr = self.fpr[self.pos_label][report_idx]
                tpr = self.tpr[self.pos_label][report_idx]
                roc_auc = self.roc_auc[self.pos_label][report_idx]

                ax.plot(
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
                    class_linestyle = LINESTYLE[(class_idx % len(LINESTYLE))]

                    ax.plot(
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

        ax.set(
            xlabel=xlabel,
            xlim=(-0.01, 1.01),
            ylabel=ylabel,
            ylim=(-0.01, 1.01),
            aspect="equal",
        )

        if plot_chance_level:
            ax.plot(
                (0, 1),
                (0, 1),
                label="Chance level (AUC = 0.5)",
                color="k",
                linestyle="--",
            )

        if despine:
            _despine_matplotlib_axis(ax)

        ax.legend(
            loc="lower right",
            title=f"{self.ml_task.title()} on $\\bf{{{self.data_source}}}$ set",
        )

    @classmethod
    def _from_predictions(
        cls,
        y_true: list[list],
        y_pred: list[list],
        *,
        estimators: list,
        estimator_names: list[str],
        ml_task,
        data_source=None,
        pos_label=None,
        drop_intermediate=True,
    ):
        estimator_classes = map(attrgetter("classes_"), estimators)
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
