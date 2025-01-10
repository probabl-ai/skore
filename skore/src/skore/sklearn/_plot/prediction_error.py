import numbers

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.validation import check_random_state

from skore.externals._sklearn_compat import _safe_indexing
from skore.sklearn._plot.utils import (
    HelpDisplayMixin,
    _despine_matplotlib_axis,
    _validate_style_kwargs,
)


class PredictionErrorDisplay(HelpDisplayMixin):
    """Visualization of the prediction error of a regression model.

    This tool can display "residuals vs predicted" or "actual vs predicted"
    using scatter plots to qualitatively assess the behavior of a regressor,
    preferably on held-out data points.

    An instance of this class is should created by
    `EstimatorReport.metrics.plot.prediction_error()`.
    You should not create an instance of this class directly.

    Parameters
    ----------
    ----------z
    y_true : ndarray of shape (n_samples,)
        True values.

    y_pred : ndarray of shape (n_samples,)
        Prediction values.

    estimator_name : str
        Name of the estimator.

    data_source : {"train", "test", "X_y"}, default=None
        The data source used to compute the ROC curve.

    Attributes
    ----------
    line_ : matplotlib Artist
        Optimal line representing `y_true == y_pred`. Therefore, it is a
        diagonal line for `kind="predictions"` and a horizontal line for
        `kind="residuals"`.

    errors_lines_ : matplotlib Artist or None
        Residual lines. If `with_errors=False`, then it is set to `None`.

    scatter_ : matplotlib Artist
        Scatter data points.

    ax_ : matplotlib Axes
        Axes with the different matplotlib axis.

    figure_ : matplotlib Figure
        Figure containing the scatter and lines.
    """

    def __init__(self, *, y_true, y_pred, estimator_name, data_source=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.estimator_name = estimator_name
        self.data_source = data_source

    def plot(
        self,
        ax=None,
        *,
        estimator_name=None,
        kind="residual_vs_predicted",
        scatter_kwargs=None,
        line_kwargs=None,
        despine=True,
    ):
        """Plot visualization.

        Extra keyword arguments will be passed to matplotlib's ``plot``.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        estimator_name : str, default=None
            Name of the estimator used to plot the prediction error. If `None`,
            we used the inferred name from the estimator.

        kind : {"actual_vs_predicted", "residual_vs_predicted"}, \
                default="residual_vs_predicted"
            The type of plot to draw:

            - "actual_vs_predicted" draws the observed values (y-axis) vs.
              the predicted values (x-axis).
            - "residual_vs_predicted" draws the residuals, i.e. difference
              between observed and predicted values, (y-axis) vs. the predicted
              values (x-axis).

        scatter_kwargs : dict, default=None
            Dictionary with keywords passed to the `matplotlib.pyplot.scatter`
            call.

        line_kwargs : dict, default=None
            Dictionary with keyword passed to the `matplotlib.pyplot.plot`
            call to draw the optimal line.

        despine : bool, default=True
            Whether to remove the top and right spines from the plot.

        Returns
        -------
        display : PredictionErrorDisplay
            Object that stores computed values.
        """
        expected_kind = ("actual_vs_predicted", "residual_vs_predicted")
        if kind not in expected_kind:
            raise ValueError(
                f"`kind` must be one of {', '.join(expected_kind)}. "
                f"Got {kind!r} instead."
            )

        if scatter_kwargs is None:
            scatter_kwargs = {}
        if line_kwargs is None:
            line_kwargs = {}

        default_scatter_kwargs = {"color": "tab:blue", "alpha": 0.8}
        default_line_kwargs = {"color": "black", "alpha": 0.7, "linestyle": "--"}

        scatter_kwargs = _validate_style_kwargs(default_scatter_kwargs, scatter_kwargs)
        line_kwargs = _validate_style_kwargs(default_line_kwargs, line_kwargs)

        scatter_kwargs = {**default_scatter_kwargs, **scatter_kwargs}
        line_kwargs = {**default_line_kwargs, **line_kwargs}

        if self.data_source in ("train", "test"):
            scatter_label = f"{self.data_source.title()} set"
        else:
            scatter_label = "Data set"

        if estimator_name is None:
            estimator_name = self.estimator_name

        if ax is None:
            _, ax = plt.subplots()

        if kind == "actual_vs_predicted":
            max_value = max(np.max(self.y_true), np.max(self.y_pred))
            min_value = min(np.min(self.y_true), np.min(self.y_pred))

            x_range = (min_value, max_value)
            y_range = (min_value, max_value)

            self.line_ = ax.plot(
                [min_value, max_value],
                [min_value, max_value],
                label="Perfect predictions",
                **line_kwargs,
            )[0]

            x_data, y_data = self.y_pred, self.y_true
            xlabel, ylabel = "Predicted values", "Actual values"

            self.scatter_ = ax.scatter(
                x_data, y_data, label=scatter_label, **scatter_kwargs
            )

            # force to have a squared axis
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks(np.linspace(min_value, max_value, num=5))
            ax.set_yticks(np.linspace(min_value, max_value, num=5))
        else:  # kind == "residual_vs_predicted"
            x_range = (np.min(self.y_pred), np.max(self.y_pred))
            residuals = self.y_true - self.y_pred
            y_range = (np.min(residuals), np.max(residuals))

            self.line_ = ax.plot(
                [np.min(self.y_pred), np.max(self.y_pred)],
                [0, 0],
                label="Perfect predictions",
                **line_kwargs,
            )[0]

            self.scatter_ = ax.scatter(
                self.y_pred, residuals, label=scatter_label, **scatter_kwargs
            )
            xlabel, ylabel = "Predicted values", "Residuals (actual - predicted)"

        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.legend(title=estimator_name)

        self.ax_ = ax
        self.figure_ = ax.figure

        if despine:
            _despine_matplotlib_axis(self.ax_, x_range=x_range, y_range=y_range)

    @classmethod
    def _from_predictions(
        cls,
        y_true,
        y_pred,
        *,
        estimator,  # currently only for consistency with other plots
        estimator_name,
        ml_task,  # FIXME: to be used when having single-output vs. multi-output
        data_source=None,
        kind="residual_vs_predicted",
        subsample=1_000,
        random_state=None,
        ax=None,
        scatter_kwargs=None,
        line_kwargs=None,
        despine=True,
    ):
        """Plot the prediction error given the true and predicted targets.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True target values.

        y_pred : array-like of shape (n_samples,)
            Predicted target values.

        estimator : estimator instance
            The estimator from which `y_pred` is obtained.

        estimator_name : str,
            The name of the estimator.

        ml_task : {"binary-classification", "multiclass-classification"}
            The machine learning task.

        data_source : {"train", "test", "X_y"}, default=None
            The data source used to compute the ROC curve.

        kind : {"actual_vs_predicted", "residual_vs_predicted"}, \
                default="residual_vs_predicted"
            The type of plot to draw:

            - "actual_vs_predicted" draws the observed values (y-axis) vs.
              the predicted values (x-axis).
            - "residual_vs_predicted" draws the residuals, i.e. difference
              between observed and predicted values, (y-axis) vs. the predicted
              values (x-axis).

        subsample : float, int or None, default=1_000
            Sampling the samples to be shown on the scatter plot. If `float`,
            it should be between 0 and 1 and represents the proportion of the
            original dataset. If `int`, it represents the number of samples
            display on the scatter plot. If `None`, no subsampling will be
            applied. by default, 1000 samples or less will be displayed.

        random_state : int or RandomState, default=None
            Controls the randomness when `subsample` is not `None`.
            See :term:`Glossary <random_state>` for details.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        scatter_kwargs : dict, default=None
            Dictionary with keywords passed to the `matplotlib.pyplot.scatter`
            call.

        line_kwargs : dict, default=None
            Dictionary with keyword passed to the `matplotlib.pyplot.plot`
            call to draw the optimal line.

        despine : bool, default=True
            Whether to remove the top and right spines from the plot.

        Returns
        -------
        display : PredictionErrorDisplay
            Object that stores the computed values.
        """
        random_state = check_random_state(random_state)

        n_samples = len(y_true)
        if isinstance(subsample, numbers.Integral):
            if subsample <= 0:
                raise ValueError(
                    f"When an integer, subsample={subsample} should be positive."
                )
        elif isinstance(subsample, numbers.Real):
            if subsample <= 0 or subsample >= 1:
                raise ValueError(
                    f"When a floating-point, subsample={subsample} should"
                    " be in the (0, 1) range."
                )
            subsample = int(n_samples * subsample)

        if subsample is not None and subsample < n_samples:
            indices = random_state.choice(np.arange(n_samples), size=subsample)
            y_true = _safe_indexing(y_true, indices, axis=0)
            y_pred = _safe_indexing(y_pred, indices, axis=0)

        viz = cls(
            y_true=y_true,
            y_pred=y_pred,
            estimator_name=estimator_name,
            data_source=data_source,
        )

        viz.plot(
            ax=ax,
            estimator_name=estimator_name,
            kind=kind,
            scatter_kwargs=scatter_kwargs,
            line_kwargs=line_kwargs,
            despine=despine,
        )

        return viz
