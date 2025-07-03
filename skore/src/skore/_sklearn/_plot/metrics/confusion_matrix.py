import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from skore._sklearn._plot.base import Display
from skore._sklearn._plot.style import StyleDisplayMixin


class ConfusionMatrixDisplay(Display):
    """Display for confusion matrix.

    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix.

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

    @StyleDisplayMixin.style_plot
    def __init__(
        self,
        confusion_matrix,
        *,
        display_labels=None,
        include_values=True,
        normalize=None,
        values_format=None,
    ):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels
        self.include_values = include_values
        self.normalize = normalize
        self.values_format = values_format
        self.figure_ = None
        self.ax_ = None
        self.text_ = None

    def plot(self, ax=None, *, cmap="Blues", colorbar=True, **kwargs):
        """Plot the confusion matrix.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If None, a new figure and axes is created.

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
        if self.normalize not in (None, "true", "pred", "all"):
            raise ValueError(
                "normalize must be one of None, 'true', 'pred', 'all'; "
                f"got {self.normalize!r}"
            )

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        n_classes = cm.shape[0]

        with np.errstate(all="ignore"):
            if self.normalize == "true":
                cm = cm / cm.sum(axis=1, keepdims=True)
            elif self.normalize == "pred":
                cm = cm / cm.sum(axis=0, keepdims=True)
            elif self.normalize == "all":
                cm = cm / cm.sum()
            else:  # None
                pass

        self.confusion_matrix = cm

        im = ax.imshow(cm, interpolation="nearest", cmap=cmap, **kwargs)
        if colorbar:
            fig.colorbar(im, ax=ax)

        if self.display_labels is None:
            self.display_labels = np.arange(n_classes)
        elif len(self.display_labels) != n_classes:
            raise ValueError(
                f"display_labels must have length equal to number of classes "
                f"({n_classes}), got {len(self.display_labels)}"
            )
        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=self.display_labels,
            yticklabels=self.display_labels,
            ylabel="True label",
            xlabel="Predicted label",
        )
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

        self.text_ = np.empty_like(cm, dtype=object)
        if self.include_values:
            fmt = self.values_format or (".2f" if self.normalize else "d")
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
    def from_predictions(
        cls,
        y_true,
        y_pred,
        *,
        sample_weight=None,
        display_labels=None,
        include_values=True,
        normalize=None,
        values_format=None,
    ):
        """Create a confusion matrix display from predictions.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True labels.

        y_pred : array-like of shape (n_samples,)
            Predicted labels, as returned by a classifier.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        display_labels : list of str, default=None
            Target names used for plotting. By default, labels will be inferred
            from y_true.

        include_values : bool, default=True
            Includes values in confusion matrix.

        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, confusion matrix will not be
            normalized.

        values_format : str, default=None
            Format specification for values in confusion matrix. If None, the format
            specification is 'd' or '.2g' whichever is shorter.

        Returns
        -------
        display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
            The confusion matrix display.
        """
        cm = sklearn_confusion_matrix(
            y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
        )

        if display_labels is None:
            display_labels = np.unique(np.concatenate([y_true, y_pred]))

        disp = cls(
            confusion_matrix=cm,
            display_labels=display_labels,
            include_values=include_values,
            normalize=normalize,
            values_format=values_format,
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
        if self.display_labels is None:
            display_labels = [f"Class {i}" for i in range(cm.shape[0])]
        else:
            display_labels = self.display_labels

        return pd.DataFrame(cm, index=display_labels, columns=display_labels)
