from typing import Any, Literal, Optional, Union

import matplotlib.pyplot as plt
from numpy.typing import NDArray

from skore.sklearn._plot.style import StyleDisplayMixin
from skore.sklearn._plot.utils import (
    HelpDisplayMixin,
    _despine_matplotlib_axis,
    _validate_style_kwargs,
)
from skore.sklearn.types import MLTask, PositiveLabel


class CalibrationCurveDisplay(StyleDisplayMixin, HelpDisplayMixin):
    """Visualization of the calibration curve for a classifier.

    A calibration curve (also known as a reliability diagram) plots the calibration
    of a classifier, showing how well the predicted probabilities match observed
    outcomes. It plots the mean predicted probability in each bin against the
    fraction of positive samples in that bin.

    An instance of this class is created by
    `EstimatorReport.metrics.calibration_curve()`. You should not create an instance
    of this class directly.

    Parameters
    ----------
    prob_true : dict[Any, list[NDArray]]
        Dictionary mapping positive labels to lists of true probabilities.

    prob_pred : dict[Any, list[NDArray]]
        Dictionary mapping positive labels to lists of predicted probabilities.

    y_prob : list[NDArray]
        List of predicted probabilities.

    estimator_names : list[str]
        List of estimator names.

    pos_label : PositiveLabel
        The positive label.

    data_source : {"train", "test", "X_y"}
        The source of the data.

    ml_task : {"binary-classification", "multiclass-classification"}
        The machine learning task.

    report_type : {"cross-validation", "estimator", "comparison-estimator"}
        The type of report.

    n_bins : int
        Number of bins used for the calibration curve.

    strategy : {"uniform", "quantile"}
        Strategy used to define the widths of the bins.

    Attributes
    ----------
    line_ : matplotlib Artist
        Calibration curve lines.

    ax_ : matplotlib Axes
        Axes with the calibration curve.

    hist_ax_ : matplotlib Axes
        Axes with the histogram of predicted probabilities.

    figure_ : matplotlib Figure
        Figure containing the curve and histogram.
    """

    _default_line_kwargs: Union[dict[str, Any], None] = None
    _default_ref_line_kwargs: Union[dict[str, Any], None] = None
    _default_hist_kwargs: Union[dict[str, Any], None] = None

    def __init__(
        self,
        *,
        prob_true: dict[Any, list[NDArray]],
        prob_pred: dict[Any, list[NDArray]],
        y_prob: list[NDArray],
        estimator_names: list[str],
        pos_label: PositiveLabel,
        data_source: Literal["train", "test", "X_y"],
        ml_task: MLTask,
        report_type: Literal["cross-validation", "estimator", "comparison-estimator"],
        n_bins: int,
        strategy: str,
    ) -> None:
        self.prob_true = prob_true
        self.prob_pred = prob_pred
        self.y_prob = y_prob
        self.estimator_names = estimator_names
        self.pos_label = pos_label
        self.data_source = data_source
        self.ml_task = ml_task
        self.report_type = report_type
        self.n_bins = n_bins
        self.strategy = strategy

    def _plot_single_estimator(
        self,
        *,
        line_kwargs: dict[str, Any],
        hist_kwargs: dict[str, Any],
    ) -> list[Any]:
        """Plot calibration curve for a single estimator.

        Parameters
        ----------
        line_kwargs : dict[str, Any]
            Keyword arguments for the line plots.

        hist_kwargs : dict[str, Any]
            Keyword arguments for the histogram.

        Returns
        -------
        lines : list[matplotlib Artist]
            The plotted lines.
        """
        lines = []
        if self.data_source in ("train", "test"):
            line_label = f"{self.data_source.title()} set"
        else:  # data_source == "X_y"
            line_label = "Data set"

        # Plot calibration curve
        cal_line = self.ax_.plot(
            self.prob_pred[self.pos_label][0],
            self.prob_true[self.pos_label][0],
            label=line_label,
            **line_kwargs,
        )[0]
        lines.append(cal_line)

        # Plot histogram of predicted probabilities
        self.hist_ax_.hist(
            self.y_prob[0],
            range=(0, 1),
            bins=self.n_bins,
            **hist_kwargs,
        )

        return lines

    def _plot_cross_validated_estimator(
        self,
        *,
        line_kwargs: dict[str, Any],
        hist_kwargs: dict[str, Any],
    ) -> list[Any]:
        """Plot calibration curve for a cross-validated estimator.

        Parameters
        ----------
        line_kwargs : dict[str, Any]
            Keyword arguments for the line plots.

        hist_kwargs : dict[str, Any]
            Keyword arguments for the histogram.

        Returns
        -------
        lines : list[matplotlib Artist]
            The plotted lines.
        """
        lines = []

        # Plot calibration curves for each fold
        for split_idx in range(len(self.y_prob)):
            label = f"Estimator of fold #{split_idx + 1}"

            cal_line = self.ax_.plot(
                self.prob_pred[self.pos_label][split_idx],
                self.prob_true[self.pos_label][split_idx],
                label=label,
                **line_kwargs,
            )[0]
            lines.append(cal_line)

            # Add to histogram
            self.hist_ax_.hist(
                self.y_prob[split_idx],
                range=(0, 1),
                bins=self.n_bins,
                **hist_kwargs,
            )

        return lines

    def _plot_comparison_estimator(
        self,
        *,
        line_kwargs: dict[str, Any],
        hist_kwargs: dict[str, Any],
    ) -> list[Any]:
        """Plot calibration curves for multiple estimators.

        Parameters
        ----------
        line_kwargs : dict[str, Any]
            Keyword arguments for the line plots.

        hist_kwargs : dict[str, Any]
            Keyword arguments for the histogram.

        Returns
        -------
        lines : list[matplotlib Artist]
            The plotted lines.
        """
        lines = []

        # Plot calibration curves for each estimator
        for estimator_idx in range(len(self.y_prob)):
            cal_line = self.ax_.plot(
                self.prob_pred[self.pos_label][estimator_idx],
                self.prob_true[self.pos_label][estimator_idx],
                label=self.estimator_names[estimator_idx],
                **line_kwargs,
            )[0]
            lines.append(cal_line)

            # Add to histogram
            self.hist_ax_.hist(
                self.y_prob[estimator_idx],
                range=(0, 1),
                bins=self.n_bins,
                **hist_kwargs,
            )

        return lines

    @StyleDisplayMixin.style_plot
    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        hist_ax: Optional[plt.Axes] = None,
        *,
        line_kwargs: Optional[dict[str, Any]] = None,
        ref_line_kwargs: Optional[dict[str, Any]] = None,
        hist_kwargs: Optional[dict[str, Any]] = None,
        despine: bool = True,
    ) -> None:
        """Plot calibration curve (line plot + histogram).

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot the calibration curve on. If `None`, a new figure and
            axes is created.

        hist_ax : matplotlib axes, default=None
            Axes object to plot the histogram on. If `None`, a new figure and axes is
            created.

        line_kwargs : dict[str, Any], default=None
            Dictionary with keywords passed to the `matplotlib.pyplot.plot` call
            for the calibration curves.

        ref_line_kwargs : dict[str, Any], default=None
            Dictionary with keywords passed to the `matplotlib.pyplot.plot` call
            for the reference line (perfectly calibrated).

        hist_kwargs : dict[str, Any], default=None
            Dictionary with keywords passed to the `matplotlib.pyplot.hist` call
            for the histogram.

        despine : bool, default=True
            Whether to remove the top and right spines from the plot.

        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = make_classification(random_state=0)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> classifier = LogisticRegression()
        >>> report = EstimatorReport(classifier, **split_data)
        >>> display = report.metrics.calibration_curve(pos_label=1)
        >>> display.plot()
        """
        # Create figure and axes if not provided
        if ax is None or hist_ax is None:
            fig, (self.ax_, self.hist_ax_) = plt.subplots(
                nrows=2, figsize=(8, 8), height_ratios=[2, 1], sharex=True
            )
            self.figure_ = fig
        else:
            self.ax_ = ax
            self.hist_ax_ = hist_ax
            self.figure_ = ax.figure

        # Set default kwargs
        default_line_kwargs = {"alpha": 0.8}
        default_ref_line_kwargs = {
            "color": "black",
            "linestyle": "--",
            "alpha": 0.8,
            "label": "Perfectly calibrated",
        }
        default_hist_kwargs = {"alpha": 0.5, "color": "gray"}

        # Update with user-provided kwargs
        line_kwargs_validated = _validate_style_kwargs(
            default_line_kwargs,
            line_kwargs or self._default_line_kwargs or {},
        )
        ref_line_kwargs_validated = _validate_style_kwargs(
            default_ref_line_kwargs,
            ref_line_kwargs or self._default_ref_line_kwargs or {},
        )
        hist_kwargs_validated = _validate_style_kwargs(
            default_hist_kwargs,
            hist_kwargs or self._default_hist_kwargs or {},
        )

        # Create plot based on report type
        if self.report_type == "estimator":
            self.line_ = self._plot_single_estimator(
                line_kwargs=line_kwargs_validated,
                hist_kwargs=hist_kwargs_validated,
            )
        elif self.report_type == "cross-validation":
            self.line_ = self._plot_cross_validated_estimator(
                line_kwargs=line_kwargs_validated,
                hist_kwargs=hist_kwargs_validated,
            )
        elif self.report_type == "comparison-estimator":
            self.line_ = self._plot_comparison_estimator(
                line_kwargs=line_kwargs_validated,
                hist_kwargs=hist_kwargs_validated,
            )
        else:
            raise ValueError(
                f"`report_type` should be one of 'estimator', 'cross-validation', "
                f"or 'comparison-estimator'. Got '{self.report_type}' instead."
            )

        # Plot reference line
        self.ax_.plot([0, 1], [0, 1], **ref_line_kwargs_validated)

        # Set labels and titles
        self.ax_.set_ylabel("Fraction of positives")
        self.ax_.set_ylim([0, 1])
        self.ax_.legend(loc="upper left")

        self.hist_ax_.set_xlabel("Mean predicted probability")
        self.hist_ax_.set_ylabel("Count")
        self.hist_ax_.set_xlim([0, 1])

        if self.data_source in ("train", "test"):
            self.ax_.set_title(
                "Calibration curve (reliability diagram) - "
                f"{self.data_source.title()} set"
            )
        else:
            self.ax_.set_title("Calibration curve (reliability diagram)")

        plt.tight_layout()

        # Despine if requested
        if despine:
            _despine_matplotlib_axis(self.ax_, x_range=(0, 1), y_range=(0, 1))
            _despine_matplotlib_axis(
                self.hist_ax_, x_range=(0, 1), y_range=self.hist_ax_.get_ylim()
            )
