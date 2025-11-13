from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn.types import YPlotData


class ConfusionMatrixDisplay(DisplayMixin):
    """Display for confusion matrix.

    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix.

    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.

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
        *,
        normalize=None,
    ):
        self.confusion_matrix = confusion_matrix
        self.normalize = normalize
        self.figure_ = None
        self.ax_ = None
        self.text_ = None

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        display_labels: list[str] | None = None,
        include_values: bool = True,
        values_format: str | None = None,
        cmap: str | Colormap = "Blues",
        colorbar: bool = True,
        **kwargs,
    ):
        """Plot the confusion matrix.

        Parameters
        ----------
        display_labels : list of str, default=None
            Display labels for plot. If None, display labels are set from 0 to
            ``n_classes - 1``.

        include_values : bool, default=True
            Includes values in confusion matrix.

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
        return self._plot(
            display_labels=display_labels,
            include_values=include_values,
            values_format=values_format,
            cmap=cmap,
            colorbar=colorbar,
        )

    def _plot_matplotlib(
        self,
        *,
        display_labels: list[str] | None = None,
        include_values: bool = True,
        values_format: str | None = None,
        cmap: str | Colormap = "Blues",
        colorbar: bool = True,
        **kwargs,
    ) -> None:
        """Matplotlib implementation of the `plot` method."""
        self.display_labels = display_labels
        self.include_values = include_values
        self.values_format = values_format

        self.figure_, self.ax_ = plt.subplots()

        cm = self.confusion_matrix
        n_classes = cm.shape[0]

        im = self.ax_.imshow(cm, interpolation="nearest", cmap=cmap, **kwargs)
        if colorbar:
            self.figure_.colorbar(im, ax=self.ax_)

        if self.display_labels is None:
            self.display_labels = np.arange(n_classes).astype(str).tolist()
        elif len(self.display_labels) != n_classes:
            raise ValueError(
                f"display_labels must have length equal to number of classes "
                f"({n_classes}), got {len(self.display_labels)}"
            )
        self.ax_.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=self.display_labels,
            yticklabels=self.display_labels,
            ylabel="True label",
            xlabel="Predicted label",
        )
        plt.setp(self.ax_.get_xticklabels(), rotation=0, ha="center")

        self.text_ = np.empty_like(cm, dtype=object)
        if self.include_values:
            fmt = self.values_format or (".2f" if self.normalize else "d")
            thresh = cm.max() / 2.0
            for i in range(n_classes):
                for j in range(n_classes):
                    txt = format(cm[i, j], fmt)
                    color = "white" if cm[i, j] > thresh else "black"
                    self.text_[i, j] = self.ax_.text(
                        j, i, txt, ha="center", va="center", color=color
                    )

        self.ax_.set_title("Confusion Matrix")
        self.figure_.tight_layout()

    @classmethod
    def _compute_data_for_display(
        cls,
        y_true: Sequence[YPlotData],
        y_pred: Sequence[YPlotData],
        normalize: Literal["true", "pred", "all"] | None = None,
        **kwargs,
    ) -> "ConfusionMatrixDisplay":
        """Compute the confusion matrix for display.

        Parameters
        ----------
        y_true : list of array-like of shape (n_samples,)
            True labels.

        y_pred : list of ndarray of shape (n_samples,)
            Predicted labels, as returned by a classifier.

        report_type : {"comparison-cross-validation", "comparison-estimator", \
                "cross-validation", "estimator"}
            The type of report.

        estimators : list of estimator instances
            The estimators from which `y_pred` is obtained.

        ml_task : {"binary-classification", "multiclass-classification"}
            The machine learning task.

        data_source : {"train", "test", "X_y"}
            The data source used to compute the ROC curve.

        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, confusion matrix will not be
            normalized.

        **kwargs : dict
            Additional keyword arguments that are ignored for compatibility with
            other metrics displays. Here, `report_type`, `estimators`, `ml_task` and
            `data_source` are ignored.

        Returns
        -------
        display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
            The confusion matrix display.
        """
        y_true_values = y_true[0].y
        y_pred_values = y_pred[0].y

        cm = sklearn_confusion_matrix(
            y_true=y_true_values,
            y_pred=y_pred_values,
            normalize=normalize,
        )
        disp = cls(
            confusion_matrix=cm,
            normalize=normalize,
        )

        return disp

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
