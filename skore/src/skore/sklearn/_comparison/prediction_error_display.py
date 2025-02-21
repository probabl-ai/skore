import numbers

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.validation import check_random_state

from skore.externals._sklearn_compat import _safe_indexing
from skore.sklearn._plot.utils import (
    HelpDisplayMixin,
    _despine_matplotlib_axis,
)


class PredictionErrorDisplay(HelpDisplayMixin):
    """Prediction error visualization for comparison report.

    This tool can display "residuals vs predicted" or "actual vs predicted"
    using scatter plots to qualitatively assess the behavior of a regressor,
    preferably on held-out data points.

    An instance of this class is should created by
    `ComparisonReport.metrics.prediction_error()`.
    You should not create an instance of this class directly.

    Parameters
    ----------
    y_true : list of ndarray of shape (n_samples,)
        True values.

    y_pred : list of ndarray of shape (n_samples,)
        Prediction values.

    estimator_names : str
        Name of the estimators.

    data_source : {"train", "test", "X_y"}
        The data source used to display the prediction error.

    Attributes
    ----------
    line_ : matplotlib line
        Optimal line representing `y_true == y_pred`. Therefore, it is a
        diagonal line for `kind="predictions"` and a horizontal line for
        `kind="residuals"`, available after calling `plot`.

    scatters_ : list of matplotlib scatters
        The scatters of the prediction error curve, available after calling `plot`.

    ax_ : matplotlib axes
        The axes on which the prediction error curve is plotted, available after calling
        `plot`.

    figure_ : matplotlib figure
        The figure on which the prediction error curve is plotted, available after
        calling `plot`.
    """

    def __init__(self, *, y_true, y_pred, estimator_names, data_source):
        self.y_true = y_true
        self.y_pred = y_pred
        self.estimator_names = estimator_names
        self.data_source = data_source

    def plot(
        self,
        ax=None,
        *,
        kind="residual_vs_predicted",
        despine=True,
    ):
        """Plot visualization.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        kind : {"actual_vs_predicted", "residual_vs_predicted"}, \
                default="residual_vs_predicted"
            The type of plot to draw:

            - "actual_vs_predicted" draws the observed values (y-axis) vs.
              the predicted values (x-axis).
            - "residual_vs_predicted" draws the residuals, i.e. difference
              between observed and predicted values, (y-axis) vs. the predicted
              values (x-axis).

        despine : bool, default=True
            Whether to remove the top and right spines from the plot.
        """
        if kind == "actual_vs_predicted":
            xlabel, ylabel = "Predicted values", "Actual values"
        elif kind == "residual_vs_predicted":
            xlabel, ylabel = "Predicted values", "Residuals (actual - predicted)"
        else:
            raise ValueError(
                "`kind` must be 'actual_vs_predicted' or 'residual_vs_predicted'. "
                f"Got {kind!r} instead."
            )

        self.figure_, self.ax_ = (ax.figure, ax) if ax else plt.subplots()
        self.scatters_ = [
            self.ax_.scatter(
                self.y_pred[report_idx],
                (
                    self.y_true[report_idx]
                    if kind == "actual_vs_predicted"
                    else (self.y_true[report_idx] - self.y_pred[report_idx])
                ),
                label=f"{report_name} #{report_idx + 1}",
                alpha=0.6,
                s=10,
            )
            for report_idx, report_name in enumerate(self.estimator_names)
        ]

        x_range_perfect_pred = [np.inf, -np.inf]
        y_range_perfect_pred = [np.inf, -np.inf]
        for y_true, y_pred in zip(self.y_true, self.y_pred):
            if kind == "actual_vs_predicted":
                min_value = min(y_pred.min(), y_true.min())
                max_value = max(y_pred.max(), y_true.max())
                x_range_perfect_pred[0] = min(x_range_perfect_pred[0], min_value)
                x_range_perfect_pred[1] = max(x_range_perfect_pred[1], max_value)
                y_range_perfect_pred[0] = min(y_range_perfect_pred[0], min_value)
                y_range_perfect_pred[1] = max(y_range_perfect_pred[1], max_value)
            else:  # kind == "residual_vs_predicted"
                residuals = y_true - y_pred
                x_range_perfect_pred[0] = min(x_range_perfect_pred[0], y_pred.min())
                x_range_perfect_pred[1] = max(x_range_perfect_pred[1], y_pred.max())
                y_range_perfect_pred[0] = min(y_range_perfect_pred[0], residuals.min())
                y_range_perfect_pred[1] = max(y_range_perfect_pred[1], residuals.max())

        self.line_ = self.ax_.plot(
            x_range_perfect_pred,
            (y_range_perfect_pred if kind == "actual_vs_predicted" else [0, 0]),
            color="black",
            alpha=0.7,
            linestyle="--",
            label="Perfect predictions",
        )[0]

        self.ax_.set(
            aspect="equal",
            xlim=x_range_perfect_pred,
            ylim=y_range_perfect_pred,
            xticks=np.linspace(x_range_perfect_pred[0], x_range_perfect_pred[1], num=5),
            yticks=np.linspace(y_range_perfect_pred[0], y_range_perfect_pred[1], num=5),
        )
        self.ax_.set(xlabel=xlabel, ylabel=ylabel)
        self.ax_.legend(title=f"Regression on $\\bf{{{self.data_source}}}$ set")

        if despine:
            _despine_matplotlib_axis(
                self.ax_,
                x_range=self.ax_.get_xlim(),
                y_range=self.ax_.get_ylim(),
            )

    @classmethod
    def _from_predictions(
        cls,
        y_true,
        y_pred,
        *,
        estimator_names,
        ml_task,
        data_source,
        subsample=1_000,
        random_state=None,
        **kwargs,
    ):
        """Private factory to create a PredictionErrorDisplay from predictions.

        Parameters
        ----------
        y_true : list of array-like of shape (n_samples,)
            True target values.

        y_pred : list of array-like of shape (n_samples,)
            Predicted target values.

        estimator_names : list[str]
            Name of the estimators used to plot the prediction error curve.

        ml_task : {"binary-classification", "multiclass-classification"}
            The machine learning task.

        data_source : {"train", "test", "X_y"}
            The data source used to compute the prediction error curve.

        subsample : float, int or None, default=1_000
            Sampling the samples to be shown on the scatter plot. If `float`,
            it should be between 0 and 1 and represents the proportion of the
            original dataset. If `int`, it represents the number of samples
            display on the scatter plot. If `None`, no subsampling will be
            applied. by default, 1000 samples or less will be displayed.

        random_state : int or RandomState, default=None
            Controls the randomness when `subsample` is not `None`.
            See :term:`Glossary <random_state>` for details.

        Returns
        -------
        display : PredictionErrorDisplay
        """
        if ml_task != "regression":
            raise ValueError("Only regression is allowed")

        random_state = check_random_state(random_state)
        if isinstance(subsample, numbers.Integral):
            if subsample <= 0:
                raise ValueError(
                    f"When an integer, subsample should be positive; got {subsample}."
                )
        elif isinstance(subsample, numbers.Real) and (subsample <= 0 or subsample >= 1):
            raise ValueError(
                f"When a floating-point, subsample should be between 0 and 1; "
                f"got {subsample}."
            )

        y_true_display, y_pred_display = [], []
        for y_true_i, y_pred_i in zip(y_true, y_pred):
            n_samples = len(y_true_i)
            if subsample is None:
                subsample_ = n_samples
            elif isinstance(subsample, numbers.Integral):
                subsample_ = subsample
            else:  # subsample is a float
                subsample_ = int(n_samples * subsample)

            # normalize subsample based on the number of splits
            subsample_ = int(subsample_ / len(y_true))
            if subsample_ < n_samples:
                indices = random_state.choice(np.arange(n_samples), size=subsample_)
                y_true_display.append(_safe_indexing(y_true_i, indices, axis=0))
                y_pred_display.append(_safe_indexing(y_pred_i, indices, axis=0))
            else:
                y_true_display.append(y_true_i)
                y_pred_display.append(y_pred_i)

        return cls(
            y_true=y_true_display,
            y_pred=y_pred_display,
            estimator_names=estimator_names,
            data_source=data_source,
        )
