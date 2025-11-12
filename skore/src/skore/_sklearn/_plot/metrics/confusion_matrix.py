from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator

from skore._sklearn._plot.base import HelpDisplayMixin, StyleDisplayMixin
from skore._sklearn.types import MLTask, ReportType, YPlotData


class ConfusionMatrixDisplay(StyleDisplayMixin, HelpDisplayMixin):
    """Display for confusion matrix.

    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix.

    Attributes
    ----------
    figure_ : matplotlib Figure
        Figure containing the confusion matrix.

    ax_ : matplotlib Axes
        Axes with confusion matrix.

    text_ : ndarray of shape (n_classes, n_classes), dtype=matplotlib Text or \
            None
        Array of matplotlib text elements containing the values in the
        confusion matrix.
    """

    def __init__(
        self,
        confusion_matrix,
    ):
        self.confusion_matrix = confusion_matrix
        self.figure_ = None
        self.ax_ = None
        self.text_ = None

    @StyleDisplayMixin.style_plot
    def plot(
        self,
        ax=None,
        *,
        display_labels=None,
        include_values=True,
        normalize=None,
        values_format=None,
        cmap="Blues",
        colorbar=True,
        **kwargs,
    ):
        """Plot the confusion matrix.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If None, a new figure and axes is created.

        display_labels : list of str, default=None
            Display labels for plot. If None, display labels are set from 0 to
            ``n_classes - 1``.

        include_values : bool, default=True
            Includes values in confusion matrix.

        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, confusion matrix will not be
            normalized.

        values_format : str, default=None
            Format specification for values in confusion matrix. If None, the format
            specification is 'd' or '.2g' whichever is shorter.

        cmap : str or matplotlib Colormap, default='Blues'
            Colormap used for confusion matrix.

        colorbar : bool, default=True
            Whether or not to add a colorbar to the plot.

        **kwargs : dict
            Additional keyword arguments to be passed to matplotlib's
            `ax.imshow`.

        Returns
        -------
        self : ConfusionMatrixDisplay
            Configured with the confusion matrix.
        """
        if display_labels is not None:
            self.display_labels = display_labels
        self.include_values = include_values
        self.normalize = normalize
        self.values_format = values_format

        if normalize not in (None, "true", "pred", "all"):
            raise ValueError(
                "normalize must be one of None, 'true', 'pred', 'all'; "
                f"got {normalize!r}"
            )

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        n_classes = cm.shape[0]

        with np.errstate(all="ignore"):
            if normalize == "true":
                cm = cm / cm.sum(axis=1, keepdims=True)
            elif normalize == "pred":
                cm = cm / cm.sum(axis=0, keepdims=True)
            elif normalize == "all":
                cm = cm / cm.sum()
            else:  # None
                pass

        self.confusion_matrix = cm

        im = ax.imshow(cm, interpolation="nearest", cmap=cmap, **kwargs)
        if colorbar:
            fig.colorbar(im, ax=ax)

        display_labels_to_use = (
            display_labels if display_labels is not None else self.display_labels
        )

        if display_labels_to_use is None:
            display_labels_to_use = np.arange(n_classes)
        elif len(display_labels_to_use) != n_classes:
            raise ValueError(
                f"display_labels must have length equal to number of classes "
                f"({n_classes}), got {len(display_labels_to_use)}"
            )
        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=display_labels_to_use,
            yticklabels=display_labels_to_use,
            ylabel="True label",
            xlabel="Predicted label",
        )
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

        self.text_ = np.empty_like(cm, dtype=object)
        if include_values:
            is_normalized = (
                np.issubdtype(cm.dtype, np.floating) or normalize is not None
            )

            fmt = values_format or ".2f" if is_normalized else values_format or "d"

            thresh = cm.max() / 2.0
            for i in range(n_classes):
                for j in range(n_classes):
                    txt = format(cm[i, j], fmt)
                    color = "white" if cm[i, j] > thresh else "black"
                    self.text_[i, j] = ax.text(
                        j, i, txt, ha="center", va="center", color=color
                    )

        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        self.figure_, self.ax_ = fig, ax
        return self

    @classmethod
    def _compute_data_for_display(
        cls,
        y_true: Sequence[YPlotData],
        y_pred: Sequence[YPlotData],
        *,
        report_type: ReportType,
        estimators: Sequence[BaseEstimator],
        ml_task: MLTask,
        data_source: Literal["train", "test", "X_y"],
        drop_intermediate: bool = True,
    ) -> "ConfusionMatrixDisplay":
        """Compute the confusion matrix for display.

        Parameters
        ----------
        y_true : list of YPlotData
            True labels.

        y_pred : list of YPlotData
            Predicted labels.

        report_type : str
            Type of report.

        estimators : list of estimators
            List of estimators.

        ml_task : str
            Machine learning task.

        data_source : str
            Data source.

        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        display : ConfusionMatrixDisplay
            Object that stores computed values.
        """
        import numpy as np
        from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

        assert len(y_true) == 1 and len(y_pred) == 1, (
            "Only single estimator is supported"
        )
        y_true_values = y_true[0].y
        y_pred_values = y_pred[0].y

        cm = sklearn_confusion_matrix(y_true=y_true_values, y_pred=y_pred_values)

        display = cls(confusion_matrix=cm)

        display.display_labels = np.unique(
            np.concatenate([y_true_values, y_pred_values])
        )

        return display

    def frame(self):
        """Return the confusion matrix as a dataframe.

        Returns
        -------
        frame : pandas.DataFrame
            The confusion matrix as a dataframe.
        """
        import pandas as pd

        cm = self.confusion_matrix
        display_labels = getattr(self, "display_labels", None)

        if display_labels is None:
            display_labels = [f"Class {i}" for i in range(cm.shape[0])]

        return pd.DataFrame(cm, index=display_labels, columns=display_labels)
