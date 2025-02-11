from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelBinarizer

from skore.sklearn._plot.utils import (
    HelpDisplayMixin,
    _ClassifierCurveDisplayMixin,
    _despine_matplotlib_axis,
    _validate_style_kwargs,
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
        pos_label=None,
        data_source=None,
    ):
        self.estimator_names = estimator_names
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc
        self.pos_label = pos_label
        self.data_source = data_source

    def plot(
        self,
        ax=None,
        *,
        estimator_names=None,
        roc_curve_kwargs=None,
        plot_chance_level=True,
        chance_level_kwargs=None,
        despine=True,
    ):
        # TO STANDARDIZE
        if ax is None:
            _, ax = plt.subplots()

        estimator_names = estimator_names or self.estimator_names
        #

        self.lines_ = []
        if len(self.fpr) == 1:  # binary-classification
            if roc_curve_kwargs is None:
                roc_curve_kwargs = [{}] * len(self.fpr[self.pos_label])
            elif isinstance(roc_curve_kwargs, dict):
                roc_curve_kwargs = [roc_curve_kwargs] * len(self.fpr[self.pos_label])
            elif isinstance(roc_curve_kwargs, list):
                if len(roc_curve_kwargs) != len(self.fpr[self.pos_label]):
                    raise ValueError(
                        "You intend to plot multiple ROC curves. We expect "
                        "`roc_curve_kwargs` to be a list of dictionaries with the "
                        "same length as the number of ROC curves. Got "
                        f"{len(roc_curve_kwargs)} instead of "
                        f"{len(self.fpr)}."
                    )
            else:
                raise ValueError(
                    "You intend to plot multiple ROC curves. We expect "
                    "`roc_curve_kwargs` to be a list of dictionaries of "
                    f"{len(self.fpr)} elements. Got {roc_curve_kwargs!r} instead."
                )

            for i in range(len(self.fpr[self.pos_label])):
                fpr = self.fpr[self.pos_label][i]
                tpr = self.tpr[self.pos_label][i]
                roc_auc = self.roc_auc[self.pos_label][i]
                estimator_name = estimator_names[i]

                default_line_kwargs = {
                    "alpha": 0.6,
                    "label": (
                        f"{self.data_source.title()} set - {estimator_name} #{i + 1} "
                        f"(AUC = {roc_auc:0.2f})"
                    ),
                }
                line_kwargs = _validate_style_kwargs(
                    default_line_kwargs, roc_curve_kwargs[i]
                )

                (line_,) = ax.plot(fpr, tpr, **line_kwargs)
                self.lines_.append(line_)

            info_pos_label = (
                f"\n(Positive label: {self.pos_label})"
                if self.pos_label is not None
                else ""
            )
        else:  # multiclass-classification
            info_pos_label = None  # irrelevant for multiclass
            colors = sample_mpl_colormap(plt.cm.tab10, len(estimator_names))

            if roc_curve_kwargs is None:
                roc_curve_kwargs = [{}] * len(self.fpr)
            elif isinstance(roc_curve_kwargs, list):
                if len(roc_curve_kwargs) != len(self.fpr):
                    raise ValueError(
                        "You intend to plot multiple ROC curves. We expect "
                        "`roc_curve_kwargs` to be a list of dictionaries with the "
                        "same length as the number of ROC curves. Got "
                        f"{len(roc_curve_kwargs)} instead of "
                        f"{len(self.fpr)}."
                    )
            else:
                raise ValueError(
                    "You intend to plot multiple ROC curves. We expect "
                    "`roc_curve_kwargs` to be a list of dictionaries of "
                    f"{len(self.fpr)} elements. Got {roc_curve_kwargs!r} instead."
                )

            for class_idx, class_ in enumerate(self.fpr):
                fpr_class = self.fpr[class_]
                tpr_class = self.tpr[class_]
                roc_auc_class = self.roc_auc[class_]
                linestyle = LINESTYLE[(class_idx % len(LINESTYLE))]

                for split_idx in range(len(fpr_class)):
                    fpr = fpr_class[split_idx]
                    tpr = tpr_class[split_idx]
                    roc_auc_mean = np.mean(roc_auc_class)
                    estimator_name = estimator_names[split_idx]

                    default_line_kwargs = {
                        "alpha": 0.6,
                        "linestyle": linestyle,
                        "color": colors[split_idx],
                        "label": (
                            f"{self.data_source.title()} set - "
                            f"class {str(class_).title()} - "
                            f"{estimator_name} #{split_idx + 1} "
                            f"(AUC = {roc_auc_mean:0.2f}"
                        ),
                    }

                    line_kwargs = _validate_style_kwargs(default_line_kwargs, {})

                    (line_,) = ax.plot(fpr, tpr, **line_kwargs)
                    self.lines_.append(line_)

        default_chance_level_line_kw = {
            "label": "Chance level (AUC = 0.5)",
            "color": "k",
            "linestyle": "--",
        }

        if chance_level_kwargs is None:
            chance_level_kwargs = {}

        chance_level_kwargs = _validate_style_kwargs(
            default_chance_level_line_kw, chance_level_kwargs
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
            (self.chance_level_,) = ax.plot((0, 1), (0, 1), **chance_level_kwargs)
        else:
            self.chance_level_ = None

        if despine:
            _despine_matplotlib_axis(ax)

        ax.legend(loc="lower right")

    @classmethod
    def _from_predictions(
        cls,
        y_true: list[list],
        y_pred: list[list],
        *,
        estimator_classes: list[list],
        estimator_names: list[str],
        ml_task,
        data_source=None,
        pos_label=None,
        drop_intermediate=True,
    ):
        fpr, tpr, roc_auc = defaultdict(list), defaultdict(list), defaultdict(list)
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
            raise ValueError

        return cls(
            fpr=fpr,
            tpr=tpr,
            roc_auc=roc_auc,
            estimator_names=estimator_names,
            pos_label=pos_label_validated,
            data_source=data_source,
        )
