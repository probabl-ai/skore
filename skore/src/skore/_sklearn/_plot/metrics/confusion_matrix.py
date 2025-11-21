from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn._plot.utils import _validate_style_kwargs
from skore._sklearn.types import ReportType, YPlotData


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
    ):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels
        self.report_type = report_type

    _default_heatmap_kwargs: dict = {
        "cmap": "Blues",
        "fmt": ".2f",
        "annot": True,
        "cbar": True,
    }

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        normalize: Literal["true", "pred", "all"] | None = None,
        heatmap_kwargs: dict | None = None,
    ):
        """Plot visualization.

        Parameters
        ----------
        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, the confusion matrix will not be
            normalized.

        heatmap_kwargs : dict, default=None
            Additional keyword arguments to be passed to seaborn's `sns.heatmap`.

        Returns
        -------
        self : ConfusionMatrixDisplay
            Configured with the confusion matrix.
        """
        return self._plot(
            normalize=normalize,
            heatmap_kwargs=heatmap_kwargs,
        )

    def _plot_matplotlib(
        self,
        *,
        normalize: Literal["true", "pred", "all"] | None = None,
        heatmap_kwargs: dict | None = None,
    ) -> None:
        """Matplotlib implementation of the `plot` method."""
        if self.report_type == "estimator":
            self._plot_single_estimator(
                normalize=normalize,
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

        heatmap_kwargs : dict, default=None
            Additional keyword arguments to be passed to seaborn's `sns.heatmap`.

        """
        self.figure_, self.ax_ = plt.subplots()

        heatmap_kwargs_validated = _validate_style_kwargs(
            self._default_heatmap_kwargs,
            heatmap_kwargs or {},
        )

        normalize_by = "normalized_by_" + normalize if normalize else "count"
        sns.heatmap(
            self.confusion_matrix.pivot(
                index="True label", columns="Predicted label", values=normalize_by
            ),
            ax=self.ax_,
            **heatmap_kwargs_validated,
        )

        self.ax_.set_title("Confusion Matrix")
        self.figure_.tight_layout()

    @classmethod
    def _compute_data_for_display(
        cls,
        y_true: Sequence[YPlotData],
        y_pred: Sequence[YPlotData],
        *,
        report_type: ReportType,
        display_labels: list[str],
        **kwargs,
    ) -> "ConfusionMatrixDisplay":
        """Compute the confusion matrix for display.

        Parameters
        ----------
        y_true : list of array-like of shape (n_samples,)
            True labels.

        y_pred : list of array-like of shape (n_samples,)
            Predicted labels, as returned by a classifier.

        report_type : {"comparison-cross-validation", "comparison-estimator", \
                "cross-validation", "estimator"}
            The type of report.

        display_labels : list of str
            Display labels for plot.

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

        cm = sklearn_confusion_matrix(
            y_true=y_true_values,
            y_pred=y_pred_values,
        )

        cm_true = cm / cm.sum(axis=1, keepdims=True)
        cm_pred = cm / cm.sum(axis=0, keepdims=True)
        cm_all = cm / cm.sum()
        n_classes = len(display_labels)

        confusion_matrix = pd.DataFrame(
            {
                "True label": np.repeat(display_labels, n_classes),
                "Predicted label": np.tile(display_labels, n_classes),
                "count": cm.flatten(),
                "normalized_by_true": cm_true.flatten(),
                "normalized_by_pred": cm_pred.flatten(),
                "normalized_by_all": cm_all.flatten(),
            }
        )

        disp = cls(
            confusion_matrix=confusion_matrix,
            display_labels=display_labels,
            report_type=report_type,
        )

        return disp

    def frame(self, normalize: Literal["true", "pred", "all"] | None = None):
        """Return the confusion matrix as a dataframe.

        Parameters
        ----------
        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, the confusion matrix will not be
            normalized.

        Returns
        -------
        frame : pandas.DataFrame
            The confusion matrix as a dataframe in pivot format with true labels as
            rows and predicted labels as columns. Values are counts or normalized
            values depending on the `normalize` parameter.
        """
        normalize_by = "normalized_by_" + normalize if normalize else "count"
        return self.confusion_matrix.pivot(
            index="True label", columns="Predicted label", values=normalize_by
        )
