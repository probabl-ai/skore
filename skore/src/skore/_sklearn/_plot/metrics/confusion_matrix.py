from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from skore._externals._sklearn_compat import confusion_matrix_at_thresholds
from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn._plot.utils import _validate_style_kwargs
from skore._sklearn.types import PositiveLabel, ReportType, YPlotData


class ConfusionMatrixDisplay(DisplayMixin):
    """Display for confusion matrix.

    Parameters
    ----------
    confusion_matrix : pd.DataFrame
        Confusion matrix data in long format with columns: "True label",
        "Predicted label", "count", "normalized_by_true", "normalized_by_pred",
        and "normalized_by_all". Each row represents one cell of the confusion matrix.

    display_labels : list of str
        Display labels for plot axes.

    report_type : {"comparison-cross-validation", "comparison-estimator", \
            "cross-validation", "estimator"}
        The type of report.

    Attributes
    ----------
    thresholds_ : list of float or None
        Thresholds of the decision function. Each threshold is associated with a
        confusion matrix. Only available for binary classification with decision scores.
        If None, the confusion matrix is the single confusion matrix for the whole
        population.

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
        do_threshold: bool = False,
    ):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels
        self.report_type = report_type
        self.do_threshold = do_threshold
        if do_threshold:
            self.thresholds_ = self.confusion_matrix["threshold"].unique()[::-1]
        else:
            self.thresholds_ = None

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
        threshold: float | list[float] | None = None,
        heatmap_kwargs: dict | None = None,
    ):
        """Plot visualization.

        Parameters
        ----------
        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, the confusion matrix will not be
            normalized.

        threshold : float, list of float, or None, default=None
            The decision threshold(s) to use for binary classification. If None,
            uses the default threshold (0.5 if thresholds are available, or the
            predicted labels if not). If a list of floats is provided, multiple
            confusion matrices will be plotted side by side.

        heatmap_kwargs : dict, default=None
            Additional keyword arguments to be passed to seaborn's `sns.heatmap`.

        Returns
        -------
        self : ConfusionMatrixDisplay
            Configured with the confusion matrix.
        """
        return self._plot(
            normalize=normalize,
            threshold=threshold,
            heatmap_kwargs=heatmap_kwargs,
        )

    def _plot_matplotlib(
        self,
        *,
        normalize: Literal["true", "pred", "all"] | None = None,
        threshold: float | list[float] | None = None,
        heatmap_kwargs: dict | None = None,
    ) -> None:
        """Matplotlib implementation of the `plot` method."""
        if self.report_type == "estimator":
            self._plot_single_estimator(
                normalize=normalize,
                threshold=threshold,
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
        threshold: float | list[float] | None = None,
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

        threshold : float, list of float, or None, default=None
            The decision threshold(s) to use for binary classification. If None,
            uses the default threshold (0.5 if thresholds are available, or the
            predicted labels if not). If a list of floats is provided, multiple
            confusion matrices will be plotted side by side.

        heatmap_kwargs : dict, default=None
            Additional keyword arguments to be passed to seaborn's `sns.heatmap`.
        """
        if threshold is not None:
            if not self.do_threshold:
                raise ValueError(
                    "threshold can only be used with binary classification and "
                    "when `report.metrics.confusion_matrix(threshold=True)` is used."
                )
            # Find the existing threshold that is closest to the given threshold
            threshold = self.thresholds_[
                np.argmin(np.abs(self.thresholds_ - threshold))
            ]
            cm = self.confusion_matrix[self.confusion_matrix["threshold"] == threshold]
        else:
            if self.do_threshold:
                # Use the default threshold (0.5) if no threshold is provided
                threshold = self.thresholds_[np.argmin(np.abs(self.thresholds_ - 0.5))]
                cm = self.confusion_matrix[
                    self.confusion_matrix["threshold"] == threshold
                ]
            else:
                cm = self.confusion_matrix

        self.figure_, self.ax_ = plt.subplots()
        heatmap_kwargs_validated = _validate_style_kwargs(
            {"fmt": ".2f" if normalize else "d", **self._default_heatmap_kwargs},
            heatmap_kwargs or {},
        )
        normalize_by = "normalized_by_" + normalize if normalize else "count"
        sns.heatmap(
            cm.pivot(
                index="true_label", columns="predicted_label", values=normalize_by
            ).reindex(index=self.display_labels, columns=self.display_labels),
            ax=self.ax_,
            **heatmap_kwargs_validated,
        )
        if threshold is not None:
            title = f"Confusion Matrix (threshold: {threshold:.2f})"
        else:
            title = "Confusion Matrix"
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
        display_labels: list[str],
        threshold: bool = False,
        pos_label: PositiveLabel | None,
        **kwargs,
    ) -> "ConfusionMatrixDisplay":
        """Compute the confusion matrix for display.

        Parameters
        ----------
        y_true : list of array-like of shape (n_samples,)
            True labels.

        y_pred : list of array-like of shape (n_samples,)
            Predicted labels or decision scores. For binary classification,
            if these are decision scores, confusion matrices at all thresholds
            will be precomputed.

        report_type : {"comparison-cross-validation", "comparison-estimator", \
                "cross-validation", "estimator"}
            The type of report.

        display_labels : list of str
            Display labels for plot.

        pos_label : int, float, bool, str or None
            The class considered as the positive class when displaying the confusion
            matrix.

        threshold : bool, default=False
            Whether to compute the confusion matrix at different thresholds.

        **kwargs : dict
            Additional keyword arguments that are ignored for compatibility with
            other metrics displays. Accepts but ignores `estimators`, `ml_task`,
            and `data_source`.

        Returns
        -------
        display : ConfusionMatrixDisplay
            The confusion matrix display.
        """
        y_true_values = y_true[0].y
        y_pred_values = y_pred[0].y
        cms = []
        if isinstance(pos_label, str):
            neg_label = next(label for label in display_labels if label != pos_label)
            display_labels = [neg_label, pos_label]

        if threshold:
            tns, fps, fns, tps, thresholds = confusion_matrix_at_thresholds(
                y_true=y_true_values,
                y_pred=y_pred_values,
                pos_label=pos_label,
            )
            for tn, fp, fn, tp in zip(tns, fps, fns, tps, strict=True):
                cms.append(np.array([[tn, fp], [fn, tp]]).astype(int))
        else:
            cms.append(
                sklearn_confusion_matrix(
                    y_true=y_true_values,
                    y_pred=y_pred_values,
                    normalize=None,  # we will normalize later
                    labels=display_labels,
                )
            )
            thresholds = [None]

        confusion_matrix_records = []
        for cm, threshold_value in zip(cms, thresholds, strict=True):
            cm_true = np.divide(
                cm,
                cm.sum(axis=1, keepdims=True),
                where=cm.sum(axis=1, keepdims=True) != 0,
            )
            cm_pred = np.divide(
                cm,
                cm.sum(axis=0, keepdims=True),
                where=cm.sum(axis=0, keepdims=True) != 0,
            )
            cm_all = np.divide(
                cm,
                cm.sum(),
                where=cm.sum() != 0,
            )

            n_classes = len(display_labels)
            true_labels = np.repeat(display_labels, n_classes)
            pred_labels = np.tile(display_labels, n_classes)

            for (
                true_label,
                pred_label,
                count,
                normalized_true,
                normalized_pred,
                normalized_all,
            ) in zip(
                true_labels,
                pred_labels,
                cm.flatten(),
                cm_true.flatten(),
                cm_pred.flatten(),
                cm_all.flatten(),
                strict=True,
            ):
                confusion_matrix_records.append(
                    {
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "count": count,
                        "normalized_by_true": normalized_true,
                        "normalized_by_pred": normalized_pred,
                        "normalized_by_all": normalized_all,
                        "threshold": threshold_value,
                    }
                )

        disp = cls(
            confusion_matrix=pd.DataFrame.from_records(confusion_matrix_records),
            display_labels=display_labels,
            report_type=report_type,
            do_threshold=threshold,
        )

        return disp

    def frame(
        self,
        normalize: Literal["true", "pred", "all"] | None = None,
        threshold: float | None = None,
    ):
        """Return the confusion matrix as a dataframe.

        Parameters
        ----------
        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, the confusion matrix will not be
            normalized.

        threshold : float or None, default=None
            The decision threshold to use for binary classification. If None and
            no thresholds are available, returns the single confusion matrix. If
            None and thresholds are available, returns all confusion matrices with
            their thresholds.

        Returns
        -------
        frame : pandas.DataFrame
            The confusion matrix as a dataframe.
        """
        cm = self.confusion_matrix
        normalize_by = "normalized_by_" + normalize if normalize else "count"

        if threshold is not None and not self.do_threshold:
            raise ValueError(
                "threshold can only be used with binary classification and "
                "when `report.metrics.confusion_matrix(threshold=True)` is used."
            )
        elif threshold is None and self.do_threshold:
            cm_columns = [
                f"{true_label}/{pred_label}"
                for true_label in self.display_labels
                for pred_label in self.display_labels
            ]

            rows = []
            for threshold_val in self.thresholds_:
                cm_at_threshold = self.frame(
                    normalize=normalize, threshold=threshold_val
                )
                flattened_values = cm_at_threshold.values.flatten()
                row = {
                    "threshold": threshold_val,
                    **dict(zip(cm_columns, flattened_values, strict=True)),
                }
                rows.append(row)

            return pd.DataFrame(rows)
        else:
            if threshold is not None:
                # Find the existing threshold that is closest to the given threshold
                threshold = self.thresholds_[
                    np.argmin(np.abs(self.thresholds_ - threshold))
                ]
                cm = cm[cm["threshold"] == threshold]
            return cm.pivot(
                index="true_label", columns="predicted_label", values=normalize_by
            ).reindex(index=self.display_labels, columns=self.display_labels)
