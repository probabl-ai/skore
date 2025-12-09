from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy._typing import NDArray
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from skore._externals._sklearn_compat import confusion_matrix_at_thresholds
from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn._plot.utils import _validate_style_kwargs
from skore._sklearn.types import MLTask, PositiveLabel, ReportType, YPlotData


class ConfusionMatrixDisplay(DisplayMixin):
    """Display for confusion matrix.

    Parameters
    ----------
    confusion_matrix : pd.DataFrame
        Confusion matrix data in long format with columns: "True label",
        "Predicted label", "count", "normalized_by_true", "normalized_by_pred",
        "normalized_by_all" and "threshold". Each row represents one cell of one
        confusion matrix.

    display_labels : list of str
        Display labels for plot axes.

    report_type : {"comparison-cross-validation", "comparison-estimator", \
            "cross-validation", "estimator"}
        The type of report.

    ml_task : {"binary-classification", "multiclass-classification"}
        The machine learning task.

    pos_label : int, float, bool, str or None
        The class considered as the positive class when displaying the confusion
        matrix.

    Attributes
    ----------
    thresholds_ : array-like of shape (n_thresholds,)
        Thresholds of the decision function. Each threshold is associated with a
        confusion matrix. Only available for binary classification.

    figure_ : matplotlib Figure
        Figure containing the confusion matrix.

    ax_ : matplotlib Axes
        Axes with confusion matrix.
    """

    def __init__(
        self,
        *,
        confusion_matrix: pd.DataFrame,
        display_labels: list[str],
        report_type: ReportType,
        ml_task: MLTask,
        thresholds: NDArray,
        pos_label: PositiveLabel,
    ):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels
        self.report_type = report_type
        self.thresholds_ = thresholds
        self.ml_task = ml_task
        self.pos_label = pos_label

    _default_heatmap_kwargs: dict = {
        "cmap": "Blues",
        "annot": True,
        "cbar": True,
    }

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        normalize: Literal["true", "pred", "all"] | None = None,
        threshold_value: float | None = None,
        heatmap_kwargs: dict | None = None,
    ):
        """Plot the confusion matrix.

        In binary classification, the confusion matrix can be displayed at various
        decision thresholds. This is useful for understanding how the model's
        predictions change as the decision threshold varies. If no threshold is
        provided, the confusion matrix is displayed at the default threshold (0.5).

        Parameters
        ----------
        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, the confusion matrix will not be
            normalized.

        threshold_value : float or None, default=None
            The decision threshold to use when applicable.
            If None and thresholds are available, plots the confusion matrix at the
            default threshold (0.5).

        heatmap_kwargs : dict, default=None
            Additional keyword arguments to be passed to seaborn's `sns.heatmap`.

        Returns
        -------
        self : ConfusionMatrixDisplay
            Configured with the confusion matrix.
        """
        return self._plot(
            normalize=normalize,
            threshold_value=threshold_value,
            heatmap_kwargs=heatmap_kwargs,
        )

    def _plot_matplotlib(
        self,
        *,
        normalize: Literal["true", "pred", "all"] | None = None,
        threshold_value: float | None = None,
        heatmap_kwargs: dict | None = None,
    ) -> None:
        """Matplotlib implementation of the `plot` method."""
        if self.report_type == "estimator":
            self._plot_single_estimator(
                normalize=normalize,
                threshold_value=threshold_value,
                heatmap_kwargs=heatmap_kwargs,
            )
        else:
            raise NotImplementedError(
                "`ConfusionMatrixDisplay` is only implemented for "
                "`EstimatorReport` for now."
            )

    def _plot_single_estimator(
        self,
        *,
        normalize: Literal["true", "pred", "all"] | None = None,
        threshold_value: float | None = None,
        heatmap_kwargs: dict | None = None,
    ) -> None:
        """
        Plot the confusion matrix for a single estimator.

        Parameters
        ----------
        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, the confusion matrix will not be
            normalized.

        threshold_value : float or None, default=None
            The decision threshold to use when applicable.
            If None and thresholds are available, plots the confusion matrix at the
            default threshold (0.5).

        heatmap_kwargs : dict, default=None
            Additional keyword arguments to be passed to seaborn's `sns.heatmap`.
        """
        self.figure_, self.ax_ = plt.subplots()

        heatmap_kwargs_validated = _validate_style_kwargs(
            {"fmt": ".2f" if normalize else "d", **self._default_heatmap_kwargs},
            heatmap_kwargs or {},
        )
        normalize_by = "normalized_by_" + normalize if normalize else "count"

        sns.heatmap(
            self.frame(threshold_value=threshold_value)
            .pivot(index="true_label", columns="predicted_label", values=normalize_by)
            .reindex(index=self.display_labels, columns=self.display_labels),
            ax=self.ax_,
            **heatmap_kwargs_validated,
        )

        title = "Confusion Matrix"
        print(f"ML task:{self.ml_task} !!")
        if self.ml_task == "binary-classification":
            if threshold_value is None:
                threshold_value = 0.5
            title = title + f"\nDecision threshold: {threshold_value:.2f}"

        if self.ml_task == "binary-classification" and self.pos_label is not None:
            ticklabels = [
                f"{label}*" if label == str(self.pos_label) else label
                for label in self.display_labels
            ]

            self.ax_.set(
                xlabel="Predicted label",
                ylabel="True label",
                title=title,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            )

            self.ax_.text(
                -0.15,
                -0.15,
                "*: the positive class",
                fontsize=9,
                style="italic",
                verticalalignment="bottom",
                horizontalalignment="left",
                transform=self.ax_.transAxes,
                bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "gray",
                },
            )
        else:
            self.ax_.set(
                xlabel="Predicted label",
                ylabel="True label",
                title=title,
            )

        self.figure_.tight_layout()

    @classmethod
    def _compute_data_for_display(
        cls,
        y_true: Sequence[YPlotData],
        y_pred: Sequence[YPlotData],
        *,
        report_type: ReportType,
        ml_task: MLTask,
        display_labels: list[str],
        pos_label: PositiveLabel,
        **kwargs,
    ) -> "ConfusionMatrixDisplay":
        """Compute the confusion matrix for display.

        Parameters
        ----------
        y_true : list of array-like of shape (n_samples,)
            True labels.

        y_pred : list of array-like of shape (n_samples,)
            Decision scores when binary classification with thresholds enabled.
            Otherwise, predicted labels.

        report_type : {"comparison-cross-validation", "comparison-estimator", \
                "cross-validation", "estimator"}
            The type of report.

        ml_task : {"binary-classification", "multiclass-classification"}
            The machine learning task.

        display_labels : list of str
            Display labels for plot.

        pos_label : int, float, bool, str or None
            The class considered as the positive class when displaying the confusion
            matrix.

        **kwargs : dict
            Additional keyword arguments that are ignored for compatibility with
            other metrics displays. Accepts but ignores `estimators` and `data_source`.

        Returns
        -------
        display : ConfusionMatrixDisplay
            The confusion matrix display.
        """
        y_true_values = y_true[0].y
        y_pred_values = y_pred[0].y

        if ml_task == "binary-classification":
            if pos_label is not None:
                neg_label = next(
                    label for label in display_labels if label != pos_label
                )
                display_labels = [str(neg_label), str(pos_label)]
            tns, fps, fns, tps, thresholds = confusion_matrix_at_thresholds(
                y_true=y_true_values,
                y_score=y_pred_values,
                pos_label=pos_label,
            )
            cms = np.column_stack([tns, fps, fns, tps]).reshape(-1, 2, 2).astype(int)
        else:
            cms = sklearn_confusion_matrix(
                y_true=y_true_values,
                y_pred=y_pred_values,
                normalize=None,  # we will normalize later
                labels=display_labels,
            )[np.newaxis, ...]
            thresholds = np.array([np.nan])

        row_sums = cms.sum(axis=2, keepdims=True)
        cm_true = np.divide(cms, row_sums, where=row_sums != 0)

        col_sums = cms.sum(axis=1, keepdims=True)
        cm_pred = np.divide(cms, col_sums, where=col_sums != 0)

        total_sums = cms.sum(axis=(1, 2), keepdims=True)
        cm_all = np.divide(cms, total_sums, where=total_sums != 0)

        n_thresholds = len(thresholds)
        n_classes = len(display_labels)
        n_cells = n_classes * n_classes

        true_labels = np.tile(np.repeat(display_labels, n_classes), n_thresholds)
        pred_labels = np.tile(np.tile(display_labels, n_classes), n_thresholds)
        threshold_values = np.repeat(thresholds, n_cells)

        counts = cms.reshape(-1)
        normalized_true_values = cm_true.reshape(-1)
        normalized_pred_values = cm_pred.reshape(-1)
        normalized_all_values = cm_all.reshape(-1)

        confusion_matrix = pd.DataFrame(
            {
                "true_label": true_labels,
                "predicted_label": pred_labels,
                "count": counts,
                "normalized_by_true": normalized_true_values,
                "normalized_by_pred": normalized_pred_values,
                "normalized_by_all": normalized_all_values,
                "threshold": threshold_values,
            }
        )

        disp = cls(
            confusion_matrix=confusion_matrix,
            display_labels=display_labels,
            report_type=report_type,
            ml_task=ml_task,
            pos_label=pos_label,
            thresholds=np.unique(thresholds),
        )

        return disp

    def frame(
        self,
        threshold_value: float | None = None,
    ):
        """Return the confusion matrix as a long format dataframe.

        In binary classification, the confusion matrix can be returned at various
        decision thresholds. This is useful for understanding how the model's
        predictions change as the decision threshold varies. If no threshold is
        provided, the default threshold (0.5) is used.

        The matrix is returned as a long format dataframe where each line represents one
        cell of the matrix. The columns are "true_label", "predicted_label", "count",
        "normalized_by_true", "normalized_by_pred", "normalized_by_all" and "threshold".

        Parameters
        ----------
        threshold_value : float or None, default=None
            The decision threshold to use when applicable.
            If None and thresholds are available, returns the confusion matrix at the
            default threshold (0.5).

        Returns
        -------
        frame : pandas.DataFrame
            The confusion matrix as a dataframe.
        """
        if threshold_value is not None and self.ml_task != "binary-classification":
            raise ValueError(
                "Threshold support is only available for binary classification."
            )
        if threshold_value is None:
            if self.ml_task == "binary-classification":
                threshold_value = 0.5
            else:
                return self.confusion_matrix

        index_right = np.searchsorted(self.thresholds_, threshold_value)
        index_left = index_right - 1
        diff_right = abs(self.thresholds_[index_right] - threshold_value)
        diff_left = abs(self.thresholds_[index_left] - threshold_value)

        threshold_value = self.thresholds_[
            index_right if diff_right < diff_left else index_left
        ]
        return self.confusion_matrix.query("threshold == @threshold_value")
