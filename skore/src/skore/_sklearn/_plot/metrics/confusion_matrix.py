from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.utils._response import _check_response_method

from skore._externals._sklearn_compat import confusion_matrix_at_thresholds
from skore._sklearn._base import BaseEstimator
from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn._plot.utils import (
    _ClassifierDisplayMixin,
    _validate_style_kwargs,
)
from skore._sklearn.types import MLTask, PositiveLabel, ReportType, YPlotData


class ConfusionMatrixDisplay(_ClassifierDisplayMixin, DisplayMixin):
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

    response_method : str
        The estimator's method that was used to get the predictions. The possible
        values are: "predict", "predict_proba", and "decision_function".

    Attributes
    ----------
    thresholds : ndarray of shape (n_thresholds,)
        Thresholds of the decision function. Each threshold is associated with a
        confusion matrix. Only available for binary classification. Thresholds are
        sorted in ascending order.

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
        response_method: str,
    ):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels
        self.report_type = report_type
        self.thresholds = thresholds
        self.ml_task = ml_task
        self.pos_label = pos_label
        self.response_method = response_method

    _default_heatmap_kwargs: dict = {
        "cmap": "Blues",
        "cbar": False,
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
        provided, the confusion matrix is displayed at the default threshold (0.5 for
        `predict_proba` response method, 0 for `decision_function` response method).

        Parameters
        ----------
        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, the confusion matrix will not be
            normalized.

        threshold_value : float or None, default=None
            The decision threshold to use when applicable.
            If None and thresholds are available, plots the confusion matrix at the
            default threshold (0.5 for `predict_proba` response method, 0 for
            `decision_function` response method).

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
            default threshold (0.5 for `predict_proba` response method, 0 for
            `decision_function` response method).

        heatmap_kwargs : dict, default=None
            Additional keyword arguments to be passed to seaborn's `sns.heatmap`.
        """
        if self.report_type not in ["estimator", "cross-validation"]:
            raise NotImplementedError(
                "`ConfusionMatrixDisplay` is only implemented for "
                "`EstimatorReport` and `CrossValidationReport` for now."
            )

        self.figure_, self.ax_ = plt.subplots()

        if self.report_type == "cross-validation":
            default_fmt = ".3f" if normalize else ".1f"
            annot_fmt = (
                heatmap_kwargs.pop("fmt", default_fmt)
                if heatmap_kwargs
                else default_fmt
            )
            aggregated = (
                self.frame(normalize=normalize, threshold_value=threshold_value)
                .groupby(["true_label", "predicted_label"])["value"]
                .agg(["mean", "std"])
                .reset_index()
            )
            aggregated["annot"] = aggregated.apply(
                lambda row: f"{row['mean']:{annot_fmt}}\n(± {row['std']:{annot_fmt}})",
                axis=1,
            )

            frame = aggregated.pivot(
                index="true_label", columns="predicted_label", values="mean"
            ).reindex(index=self.display_labels, columns=self.display_labels)
            annot = aggregated.pivot(
                index="true_label", columns="predicted_label", values="annot"
            ).reindex(index=self.display_labels, columns=self.display_labels)
            default_fmt = ""
        else:
            frame = (
                self.frame(normalize=normalize, threshold_value=threshold_value)
                .pivot(index="true_label", columns="predicted_label", values="value")
                .reindex(index=self.display_labels, columns=self.display_labels)
            )
            annot = True
            default_fmt = ".2f" if normalize else "d"

        heatmap_kwargs_validated = _validate_style_kwargs(
            {"fmt": default_fmt, "annot": annot, **self._default_heatmap_kwargs},
            heatmap_kwargs or {},
        )

        sns.heatmap(
            frame,
            ax=self.ax_,
            **heatmap_kwargs_validated,
        )

        title = "Confusion Matrix"
        if self.ml_task == "binary-classification":
            if threshold_value is None:
                threshold_value = 0.5 if self.response_method == "predict_proba" else 0
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
        estimators: Sequence[BaseEstimator],
        ml_task: MLTask,
        display_labels: list[str],
        pos_label: PositiveLabel,
        response_method: str | list[str] | tuple[str, ...],
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

        estimators : list of BaseEstimator
            The estimators.

        ml_task : {"binary-classification", "multiclass-classification"}
            The machine learning task.

        display_labels : list of str
            Display labels for plot.

        pos_label : int, float, bool, str or None
            The class considered as the positive class when displaying the confusion
            matrix.

        response_method : str or list of str or tuple of str
            The estimator's method to be invoked to get the predictions. The possible
            values are: `predict`, `predict_proba`, `predict_log_proba`, and
            `decision_function`.

        **kwargs : dict
            Additional keyword arguments that are ignored for compatibility with
            other metrics displays. Accepts but ignores `estimators` and `data_source`.

        Returns
        -------
        display : ConfusionMatrixDisplay
            The confusion matrix display.
        """
        pos_label_validated = cls._validate_from_predictions_params(
            y_true, y_pred, ml_task=ml_task, pos_label=pos_label
        )

        # When provided, the positive label is set in second position.
        if ml_task == "binary-classification" and pos_label_validated is not None:
            neg_label = next(
                label for label in display_labels if label != pos_label_validated
            )
            display_labels = [str(neg_label), str(pos_label_validated)]

        cm_records = []
        for y_true_i, y_pred_i in zip(y_true, y_pred, strict=False):
            if ml_task == "binary-classification":
                tns, fps, fns, tps, thresholds = confusion_matrix_at_thresholds(
                    y_true=y_true_i.y,
                    y_score=y_pred_i.y,
                    pos_label=pos_label_validated,
                )
                cms = (
                    np.column_stack([tns, fps, fns, tps]).reshape(-1, 2, 2).astype(int)
                )
            else:
                cms = sklearn_confusion_matrix(
                    y_true=y_true_i.y,
                    y_pred=y_pred_i.y,
                    normalize=None,  # we will normalize later
                    labels=display_labels,
                )[np.newaxis, ...]
                thresholds = np.array([np.nan])

            counts = cms.reshape(-1)

            row_sums = cms.sum(axis=2, keepdims=True)
            cm_true = np.zeros_like(cms, dtype=float)
            np.divide(cms, row_sums, out=cm_true, where=row_sums != 0)
            normalized_true_values = cm_true.reshape(-1)

            col_sums = cms.sum(axis=1, keepdims=True)
            cm_pred = np.zeros_like(cms, dtype=float)
            np.divide(cms, col_sums, out=cm_pred, where=col_sums != 0)
            normalized_pred_values = cm_pred.reshape(-1)

            total_sums = cms.sum(axis=(1, 2), keepdims=True)
            cm_all = np.zeros_like(cms, dtype=float)
            np.divide(cms, total_sums, out=cm_all, where=total_sums != 0)
            normalized_all_values = cm_all.reshape(-1)

            n_thresholds = len(thresholds)
            n_classes = len(display_labels)
            n_cells = n_classes * n_classes

            true_labels = np.tile(np.repeat(display_labels, n_classes), n_thresholds)
            pred_labels = np.tile(np.tile(display_labels, n_classes), n_thresholds)
            threshold_values = np.repeat(thresholds, n_cells)

            cm_records.append(
                pd.DataFrame(
                    {
                        "true_label": true_labels,
                        "predicted_label": pred_labels,
                        "count": counts,
                        "normalized_by_true": normalized_true_values,
                        "normalized_by_pred": normalized_pred_values,
                        "normalized_by_all": normalized_all_values,
                        "threshold": threshold_values,
                        "split": y_true_i.split,
                    }
                )
            )
        confusion_matrix = pd.concat(cm_records)
        disp = cls(
            confusion_matrix=confusion_matrix,
            display_labels=display_labels,
            report_type=report_type,
            ml_task=ml_task,
            pos_label=pos_label_validated,
            response_method=_check_response_method(
                estimators[0], response_method
            ).__name__,
            thresholds=np.unique(confusion_matrix["threshold"]),
        )

        return disp

    def frame(
        self,
        *,
        normalize: Literal["true", "pred", "all"] | None = None,
        threshold_value: float | None = None,
    ):
        """Return the confusion matrix as a long format dataframe.

        In binary classification, the confusion matrix can be returned at various
        decision thresholds. This is useful for understanding how the model's
        predictions change as the decision threshold varies. If no threshold is
        provided, the default threshold (0.5 for `predict_proba` response method, 0 for
        `decision_function` response method) is used.

        The matrix is returned as a long format dataframe where each line represents one
        cell of the matrix. The columns are "true_label", "predicted_label", "value"
        and "threshold", where "value" is one of {"count", "normalized_by_true",
        "normalized_by_pred", "normalized_by_all"}, depending on the value of
        `normalize`.

        Parameters
        ----------
        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, the confusion matrix will not be
            normalized.

        threshold_value : float or None, default=None
            The decision threshold to use when applicable.
            If None and thresholds are available, returns the confusion matrix at the
            default threshold (0.5 for `predict_proba` response method, 0 for
            `decision_function` response method).

        Returns
        -------
        frame : pandas.DataFrame
            The confusion matrix as a dataframe.
        """
        normalize_col = "normalized_by_" + normalize if normalize else "count"
        if threshold_value is not None and self.ml_task != "binary-classification":
            raise ValueError(
                "Threshold support is only available for binary classification."
            )
        if threshold_value is None:
            if self.ml_task == "binary-classification":
                threshold_value = 0.5 if self.response_method == "predict_proba" else 0
            else:
                return self.confusion_matrix[
                    [
                        "true_label",
                        "predicted_label",
                        normalize_col,
                        "threshold",
                        "split",
                    ]
                ].rename(columns={normalize_col: "value"})

        frames = []
        splits = (
            self.confusion_matrix["split"].unique()
            if self.report_type == "cross-validation"
            else [None]
        )

        # Thresholding is different for each split.
        for split in splits:
            if split is None:
                frame = self.confusion_matrix
            else:
                frame = self.confusion_matrix.query("split == @split")
            thresholds = np.sort(frame["threshold"].unique())
            index_right = int(np.searchsorted(thresholds, threshold_value))
            if index_right == len(thresholds):
                index_right = index_right - 1
            elif index_right == 0 and len(thresholds) > 1:
                index_right = 1
            index_left = index_right - 1
            diff_right = abs(thresholds[index_right] - threshold_value)
            diff_left = abs(thresholds[index_left] - threshold_value)
            closest_threshold_value = thresholds[
                index_right if diff_right < diff_left else index_left
            ]
            frame = frame.query(f"threshold == {closest_threshold_value}")
            frame = frame[
                ["true_label", "predicted_label", normalize_col, "threshold", "split"]
            ].rename(columns={normalize_col: "value"})
            frames.append(frame)

        return pd.concat(frames)
