from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn.types import ReportType, YPlotData


class ConfusionMatrixDisplay(DisplayMixin):
    """Display for confusion matrix.

    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix.

    display_labels : list of str, default=None
        Display labels for plot. If None, display labels are set from 0 to
        ``n_classes - 1``.

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
        *,
        confusion_matrix: NDArray,
        display_labels: list[str] | None = None,
        normalize: Literal["true", "pred", "all"] | None = None,
        report_type: ReportType,
    ):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels
        self.normalize = normalize
        self.report_type = report_type

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        include_values: bool = True,
        values_format: str | None = None,
        cmap: str | Colormap = "Blues",
        colorbar: bool = True,
        **kwargs,
    ):
        """Plot visualization.

        Parameters
        ----------
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
            include_values=include_values,
            values_format=values_format,
            cmap=cmap,
            colorbar=colorbar,
        )

    def _plot_matplotlib(
        self,
        *,
        include_values: bool = True,
        values_format: str | None = None,
        cmap: str | Colormap = "Blues",
        colorbar: bool = True,
        **kwargs,
    ) -> None:
        """Matplotlib implementation of the `plot` method."""
        if self.report_type == "estimator":
            self._plot_single_estimator(
                include_values=include_values,
                values_format=values_format,
                cmap=cmap,
                colorbar=colorbar,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                "`ConfusionMatrixDisplay` is only implemented for"
                "`EstimatorReport` for now."
            )

    def _plot_single_estimator(
        self,
        *,
        include_values: bool = True,
        values_format: str | None = None,
        cmap: str | Colormap = "Blues",
        colorbar: bool = True,
        **kwargs,
    ) -> None:
        """
        Plot the confusion matrix for a single estimator.

        Parameters
        ----------
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
        """
        self.include_values = include_values
        self.values_format = values_format

        self.figure_, self.ax_ = plt.subplots()

        cm = self.confusion_matrix
        n_classes = cm.shape[0]

        im = self.ax_.imshow(cm, interpolation="nearest", cmap=cmap, **kwargs)
        if colorbar:
            self.figure_.colorbar(im, ax=self.ax_)

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
        *,
        report_type: ReportType,
        display_labels: list[str] | None = None,
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

        display_labels : list of str, default=None
            Display labels for plot. If None, display labels are set from 0 to
            ``n_classes - 1``.

        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, confusion matrix will not be
            normalized.

        **kwargs : dict
            Additional keyword arguments that are ignored for compatibility with
            other metrics displays. Here, `estimators`, `ml_task` and
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

        n_classes = cm.shape[0]
        if display_labels is None:
            display_labels = (
                np.unique(np.concat([y_true_values, y_pred_values]))
                .astype(str)
                .tolist()
            )
        elif len(display_labels) != n_classes:
            raise ValueError(
                f"display_labels must have length equal to number of classes "
                f"({n_classes}), got {len(display_labels)}"
            )

        disp = cls(
            confusion_matrix=cm,
            report_type=report_type,
            display_labels=display_labels,
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

        return pd.DataFrame(cm, index=display_labels, columns=display_labels)
