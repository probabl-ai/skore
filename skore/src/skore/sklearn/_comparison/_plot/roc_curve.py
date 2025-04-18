"""RocCurveDisplay variant for ComparisonReport of CrossValidationReports."""

from typing import Any, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike
from skore.sklearn._plot.style import StyleDisplayMixin
from skore.sklearn._plot.utils import (
    HelpDisplayMixin,
    _ClassifierCurveDisplayMixin,
    _despine_matplotlib_axis,
    _validate_style_kwargs,
)
from skore.sklearn.types import MLTask, PositiveLabel


class CompCVRocCurveDisplay(
    StyleDisplayMixin, HelpDisplayMixin, _ClassifierCurveDisplayMixin
):
    """RocCurveDisplay variant for ComparisonReport of CrossValidationReports."""

    _default_roc_curve_kwargs: Union[dict[str, Any], None] = None
    _default_chance_level_kwargs: Union[dict[str, Any], None] = None

    def __init__(
        self,
        fpr: list[dict[PositiveLabel, list[ArrayLike]]],
        tpr: list[dict[PositiveLabel, list[ArrayLike]]],
        roc_auc: list[dict[PositiveLabel, list[float]]],
        estimator_names: list[str],
        pos_label: Optional[PositiveLabel],
        data_source: Literal["train", "test", "X_y"],
        ml_task: MLTask,
    ):
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc
        self.estimator_names = estimator_names
        self.pos_label = pos_label
        self.data_source = data_source
        self.ml_task = ml_task

    @staticmethod
    def _plot(
        *,
        fpr: list[dict[PositiveLabel, list[ArrayLike]]],
        tpr: list[dict[PositiveLabel, list[ArrayLike]]],
        roc_auc: list[dict[PositiveLabel, list[float]]],
        pos_label: PositiveLabel,
        data_source: Literal["train", "test", "X_y"],
        ml_task: MLTask,
        ax: Axes,
        estimator_names: list[str],
        roc_curve_kwargs: list[dict[str, Any]],
    ):
        """Add lines to ax."""
        lines: list[Line2D] = []
        line_kwargs: dict[str, Any] = {"alpha": 0.5}

        if ml_task == "binary-classification":
            for report_idx, (fpr_cv, tpr_cv, roc_auc_cv, est_name) in enumerate(
                zip(fpr, tpr, roc_auc, estimator_names)
            ):
                roc_auc_mean = np.mean(roc_auc_cv[pos_label])
                roc_auc_std = np.std(roc_auc_cv[pos_label], mean=roc_auc_mean)

                # line_kwargs_validated = _validate_style_kwargs(
                #     line_kwargs, roc_curve_kwargs[report_idx]
                # )
                line_kwargs_validated = line_kwargs
                line_kwargs_validated["label"] = (
                    f"{est_name} (AUC = {roc_auc_mean:0.2f} +/- {roc_auc_std:0.2f})"
                )

                segments = [
                    list(zip(f, t))
                    for f, t in zip(fpr_cv[pos_label], tpr_cv[pos_label])
                ]
                line_collection = LineCollection(
                    segments,
                    **line_kwargs_validated,
                    color=plt.get_cmap("tab10")(report_idx % 10),
                )
                lines.append(line_collection)
                ax.add_collection(line_collection)

            info_pos_label = (
                f"\n(Positive label: {pos_label})" if pos_label is not None else ""
            )
        else:  # multiclass-classification
            raise NotImplementedError()

        ax.legend(
            bbox_to_anchor=(1.02, 1),
            title=f"{ml_task.title()} on $\\bf{{{data_source}}}$ set",
        )

        return ax, lines, info_pos_label

    @StyleDisplayMixin.style_plot
    def plot(
        self,
        ax: Optional[Axes] = None,
        *,
        roc_curve_kwargs: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None,
        plot_chance_level: bool = True,
        chance_level_kwargs: Optional[dict[str, Any]] = None,
        despine: bool = True,
    ) -> None:
        """Plot the curve."""
        self.figure_, self.ax_ = (ax.figure, ax) if ax is not None else plt.subplots()

        if roc_curve_kwargs is None:
            # roc_curve_kwargs = self._default_roc_curve_kwargs
            roc_curve_kwargs = [{}]
        # roc_curve_kwargs = self._validate_curve_kwargs(
        #     curve_param_name="roc_curve_kwargs",
        #     curve_kwargs=roc_curve_kwargs,
        #     metric=self.roc_auc,
        #     report_type=self.report_type,
        # )

        self.ax_, self.lines_, info_pos_label = CompCVRocCurveDisplay._plot(
            ax=self.ax_,
            fpr=self.fpr,
            tpr=self.tpr,
            roc_auc=self.roc_auc,
            pos_label=self.pos_label,
            data_source=self.data_source,
            ml_task=self.ml_task,
            estimator_names=self.estimator_names,
            roc_curve_kwargs=roc_curve_kwargs,
        )

        chance_level_kwargs = _validate_style_kwargs(
            {
                "label": "Chance level (AUC = 0.5)",
                "color": "k",
                "linestyle": "--",
            },
            chance_level_kwargs or self._default_chance_level_kwargs or {},
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

        self.chance_level_: Optional[Line2D] = None
        if plot_chance_level:
            (self.chance_level_,) = self.ax_.plot((0, 1), (0, 1), **chance_level_kwargs)

        if despine:
            _despine_matplotlib_axis(self.ax_)
