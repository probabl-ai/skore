from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn._plot.utils import _validate_style_kwargs
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

    report_type : {"comparison-cross-validation", "comparison-estimator", \
            "cross-validation", "estimator"}
        The type of report.

    Attributes
    ----------
    figure_ : matplotlib Figure
        Figure containing the confusion matrix.

    ax_ : matplotlib Axes
        Axes with confusion matrix.

    text_ : ndarray of shape (n_classes, n_classes) or None
        Array of matplotlib Text objects containing the values in the confusion
        matrix when `include_values=True` in the `plot()` method. Set to None
        when `include_values=False`.
    """

    def __init__(
        self,
        *,
        confusion_matrix_data: pd.DataFrame,
        display_labels: list[str] | None = None,
        report_type: ReportType,
    ):
        self.confusion_matrix_data = confusion_matrix_data
        self.display_labels = display_labels
        self.report_type = report_type

    _default_imshow_kwargs: dict | None = None
    _default_text_kwargs: dict | None = None

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        include_values: bool = True,
        colorbar: bool = True,
        normalize: Literal["true", "pred", "all"] | None = None,
        imshow_kwargs: dict | None = None,
        text_kwargs: dict | None = None,
    ):
        """Plot visualization.

        Parameters
        ----------
        include_values : bool, default=True
            Includes values in confusion matrix.

        colorbar : bool, default=True
            Whether or not to add a colorbar to the plot.

        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, the confusion matrix will not be
            normalized.

        imshow_kwargs : dict, default=None
            Additional keyword arguments to be passed to matplotlib's
            `ax.imshow`.

        text_kwargs : dict, default=None
            Additional keyword arguments to be passed to matplotlib's
            `ax.text`. Can include a 'values_format' key to specify the
            format specification for values in confusion matrix. If None,
            the format specification is 'd' or '.2f' whichever is shorter.

        Returns
        -------
        self : ConfusionMatrixDisplay
            Configured with the confusion matrix.
        """
        return self._plot(
            include_values=include_values,
            colorbar=colorbar,
            normalize=normalize,
            imshow_kwargs=imshow_kwargs,
            text_kwargs=text_kwargs,
        )

    def _plot_matplotlib(
        self,
        *,
        include_values: bool = True,
        colorbar: bool = True,
        normalize: Literal["true", "pred", "all"] | None = None,
        imshow_kwargs: dict | None = None,
        text_kwargs: dict | None = None,
    ) -> None:
        """Matplotlib implementation of the `plot` method."""
        if self.report_type == "estimator":
            self._plot_single_estimator(
                include_values=include_values,
                colorbar=colorbar,
                normalize=normalize,
                imshow_kwargs=imshow_kwargs,
                text_kwargs=text_kwargs,
            )
        else:
            raise NotImplementedError(
                "`ConfusionMatrixDisplay` is only implemented for "
                "`EstimatorReport` for now."
            )

    def _plot_single_estimator(
        self,
        *,
        include_values: bool = True,
        colorbar: bool = True,
        normalize: Literal["true", "pred", "all"] | None = None,
        imshow_kwargs: dict | None = None,
        text_kwargs: dict | None = None,
    ) -> None:
        """
        Plot the confusion matrix for a single estimator.

        Parameters
        ----------
        include_values : bool, default=True
            Includes values in confusion matrix.

        colorbar : bool, default=True
            Whether or not to add a colorbar to the plot.

        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, the confusion matrix will not be
            normalized.

        imshow_kwargs : dict, default=None
            Additional keyword arguments to be passed to matplotlib's
            `ax.imshow`.

        text_kwargs : dict, default=None
            Additional keyword arguments to be passed to matplotlib's
            `ax.text`. Can include a 'values_format' key to specify the
            format specification for values in confusion matrix. If None,
            the format specification is 'd' if `normalize` is None,
            otherwise '.2f'.
        """
        self.text_: NDArray | None = None
        self.figure_, self.ax_ = plt.subplots()

        n_classes = np.sqrt(len(self.confusion_matrix_data))
        cm = self.confusion_matrix_data.value.values
        if normalize == "true":
            cm = cm / self.confusion_matrix_data.true_sum.values
        elif normalize == "pred":
            cm = cm / self.confusion_matrix_data.pred_sum.values
        elif normalize == "all":
            cm = cm / self.confusion_matrix_data.total_sum.values

        imshow_kwargs_validated = _validate_style_kwargs(
            {"cmap": "Blues"},
            imshow_kwargs or self._default_imshow_kwargs or {},
        )

        im = self.ax_.imshow(
            cm.reshape(n_classes, n_classes),
            interpolation="nearest",
            **imshow_kwargs_validated,
        )
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

        if include_values:
            cm = cm.reshape(n_classes, n_classes)
            text_kwargs_validated = _validate_style_kwargs(
                {},
                text_kwargs or self._default_text_kwargs or {},
            )
            values_format = text_kwargs_validated.pop("values_format", None)

            self.text_ = np.empty((n_classes, n_classes), dtype=object)
            fmt = values_format or (".2f" if normalize else "d")
            thresh = cm.max() / 2.0
            for i in range(n_classes):
                for j in range(n_classes):
                    txt = format(cm[i, j], fmt)
                    color = "white" if cm[i, j] > thresh else "black"
                    self.text_[i, j] = self.ax_.text(
                        j,
                        i,
                        txt,
                        ha="center",
                        va="center",
                        color=color,
                        **text_kwargs_validated,
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

        estimators : list of estimator instances
            The estimators from which `y_pred` is obtained.

        ml_task : {"binary-classification", "multiclass-classification"}
            The machine learning task.

        data_source : {"train", "test", "X_y"}
            The data source used to compute the ROC curve.

        display_labels : list of str, default=None
            Display labels for plot. If None, display labels are set from 0 to
            ``n_classes - 1``.

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

        confusion_matrix_data = pd.DataFrame(
            {
                "true_label": np.repeat(display_labels, n_classes),
                "predicted_label": np.tile(display_labels, n_classes),
                "value": cm.ravel(),
                "true_sum": np.repeat(cm.sum(axis=1), n_classes),
                "pred_sum": np.tile(cm.sum(axis=0), n_classes),
                "total_sum": np.full(n_classes * n_classes, cm.sum()),
            }
        )

        disp = cls(
            confusion_matrix_data=confusion_matrix_data,
            report_type=report_type,
            display_labels=display_labels,
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
            The confusion matrix as a dataframe.
        """
        n_classes = np.sqrt(len(self.confusion_matrix_data))
        cm = self.confusion_matrix_data.value.values
        if normalize == "true":
            cm = cm / self.confusion_matrix_data.true_sum.values
        elif normalize == "pred":
            cm = cm / self.confusion_matrix_data.pred_sum.values
        elif normalize == "all":
            cm = cm / self.confusion_matrix_data.total_sum.values

        return pd.DataFrame(
            cm.reshape(n_classes, n_classes),
            index=self.display_labels,
            columns=self.display_labels,
        )
