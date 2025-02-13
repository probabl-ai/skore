from collections import defaultdict
from operator import attrgetter

import matplotlib.pyplot as plt
import numpy as np
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
    def __init__(
        self,
        precision,
        recall,
        *,
        average_precision,
        estimator_names,
        ml_task,
        pos_label=None,
        data_source=None,
    ):
        self.precision = precision
        self.recall = recall
        self.average_precision = average_precision
        self.estimator_names = estimator_names
        self.ml_task = ml_task
        self.pos_label = pos_label
        self.data_source = data_source

    def plot(
        self,
        ax=None,
        *,
        despine=True,
    ):
        if ax is None:
            _, ax = plt.subplots()

        self.lines_ = []
        self.chance_levels_ = []

        if self.ml_task == "binary-classification":
            for report_idx, report_name in enumerate(self.estimator_names):
                precision = self.precision[self.pos_label][report_idx]
                recall = self.recall[self.pos_label][report_idx]
                average_precision = self.average_precision[self.pos_label][report_idx]

                (line_,) = ax.plot(
                    recall,
                    precision,
                    drawstyle="steps-post",
                    alpha=0.6,
                    label=(
                        f"{report_name} #{report_idx + 1} "
                        f"(AP = {average_precision:0.2f})"
                    ),
                )

                self.lines_.append(line_)

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

                for class_idx, class_ in enumerate(self.precision):
                    precision = self.precision[class_][report_idx]
                    recall = self.recall[class_][report_idx]
                    average_precision_class = self.average_precision[class_]
                    average_precision = average_precision_class[report_idx]
                    class_linestyle = LINESTYLE[(class_idx % len(LINESTYLE))]

                    (line_,) = ax.plot(
                        recall,
                        precision,
                        color=report_color,
                        linestyle=class_linestyle,
                        alpha=0.6,
                        label=(
                            f"{report_name} #{report_idx + 1} - class {str(class_)} "
                            f"(AP = {np.mean(average_precision_class):0.2f})"
                        ),
                    )

                    self.lines_.append(line_)

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
            raise ValueError

        return cls(
            precision=precision,
            recall=recall,
            average_precision=average_precision,
            estimator_names=estimator_names,
            ml_task=ml_task,
            pos_label=pos_label_validated,
            data_source=data_source,
        )
