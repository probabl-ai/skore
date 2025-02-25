import numbers
from typing import Any, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import _num_samples, check_array, check_random_state

from skore.externals._sklearn_compat import _safe_indexing
from skore.sklearn._plot.style import StyleDisplayMixin
from skore.sklearn._plot.utils import (
    HelpDisplayMixin,
    _despine_matplotlib_axis,
    _validate_style_kwargs,
    sample_mpl_colormap,
)
from skore.sklearn.types import MLTask


class PredictionErrorDisplay(HelpDisplayMixin, StyleDisplayMixin):
    """Visualization of the prediction error of a regression model.

    This tool can display "residuals vs predicted" or "actual vs predicted"
    using scatter plots to qualitatively assess the behavior of a regressor,
    preferably on held-out data points.

    An instance of this class is should created by
    `EstimatorReport.metrics.prediction_error()`.
    You should not create an instance of this class directly.

    Parameters
    ----------
    y_true : list of ndarray of shape (n_samples,)
        True values.

    y_pred : list of ndarray of shape (n_samples,)
        Prediction values.

    estimator_name : str
        Name of the estimator.

    data_source : {"train", "test", "X_y"}
        The data source used to display the prediction error.

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

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.model_selection import train_test_split
    >>> from skore import EstimatorReport
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     *load_diabetes(return_X_y=True), random_state=0
    ... )
    >>> classifier = Ridge()
    >>> report = EstimatorReport(
    ...     classifier,
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     X_test=X_test,
    ...     y_test=y_test,
    ... )
    >>> display = report.metrics.prediction_error()
    >>> display.plot(kind="actual_vs_predicted")
    """

    _default_scatter_kwargs: Union[dict[str, Any], None] = None
    _default_line_kwargs: Union[dict[str, Any], None] = None

    def __init__(
        self,
        *,
        y_true: list[NDArray],
        y_pred: list[NDArray],
        estimator_name: str,
        data_source: Literal["train", "test", "X_y"],
    ) -> None:
        self.y_true = y_true
        self.y_pred = y_pred
        self.estimator_name = estimator_name
        self.data_source = data_source

    def plot(
        self,
        ax: Optional[Axes] = None,
        *,
        estimator_name: Optional[str] = None,
        kind: Literal[
            "actual_vs_predicted", "residual_vs_predicted"
        ] = "residual_vs_predicted",
        scatter_kwargs: Optional[dict[str, Any]] = None,
        line_kwargs: Optional[dict[str, Any]] = None,
        despine: bool = True,
    ) -> None:
        """Plot visualization.

        Extra keyword arguments will be passed to matplotlib's ``plot``.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        estimator_name : str
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

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     *load_diabetes(return_X_y=True), random_state=0
        ... )
        >>> classifier = Ridge()
        >>> report = EstimatorReport(
        ...     classifier,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
        >>> display = report.metrics.prediction_error()
        >>> display.plot(kind="actual_vs_predicted")
        """
        expected_kind = ("actual_vs_predicted", "residual_vs_predicted")
        if kind not in expected_kind:
            raise ValueError(
                f"`kind` must be one of {', '.join(expected_kind)}. "
                f"Got {kind!r} instead."
            )
        if kind == "actual_vs_predicted":
            xlabel, ylabel = "Predicted values", "Actual values"
        else:  # kind == "residual_vs_predicted"
            xlabel, ylabel = "Predicted values", "Residuals (actual - predicted)"

        scatter_kwargs = (
            self._default_scatter_kwargs if scatter_kwargs is None else scatter_kwargs
        ) or {}
        line_kwargs = (
            self._default_line_kwargs if line_kwargs is None else line_kwargs
        ) or {}

        if estimator_name is None:
            estimator_name = self.estimator_name

        if ax is None:
            _, ax = plt.subplots()

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
            else:
                residuals = y_true - y_pred
                x_range_perfect_pred[0] = min(x_range_perfect_pred[0], y_pred.min())
                x_range_perfect_pred[1] = max(x_range_perfect_pred[1], y_pred.max())
                y_range_perfect_pred[0] = min(y_range_perfect_pred[0], residuals.min())
                y_range_perfect_pred[1] = max(y_range_perfect_pred[1], residuals.max())

        colors_markers = sample_mpl_colormap(
            colormaps.get_cmap("tab10"),
            len(self.y_true) if len(self.y_true) > 10 else 10,
        )
        for split_idx in range(len(self.y_true)):
            y_true, y_pred = self.y_true[split_idx], self.y_pred[split_idx]

            default_scatter_kwargs = {
                "color": colors_markers[split_idx],
                "alpha": 0.3,
                "s": 10,
            }
            prediction_error_scatter_kwargs = _validate_style_kwargs(
                default_scatter_kwargs, scatter_kwargs
            )

            if self.data_source in ("train", "test"):
                scatter_label = f"{self.data_source.title()} set"
            else:  # data_source == "X_y"
                scatter_label = "Data set"

            if len(self.y_true) > 1:  # cross-validation
                scatter_label += f" - split #{split_idx + 1}"

            if kind == "actual_vs_predicted":
                self.scatter_ = ax.scatter(
                    y_pred,
                    y_true,
                    label=scatter_label,
                    **prediction_error_scatter_kwargs,
                )

            else:  # kind == "residual_vs_predicted"
                residuals = y_true - y_pred
                self.scatter_ = ax.scatter(
                    y_pred,
                    residuals,
                    label=scatter_label,
                    **prediction_error_scatter_kwargs,
                )

        default_line_kwargs = {
            "color": "black",
            "alpha": 0.7,
            "linestyle": "--",
            "label": "Perfect predictions",
        }
        perfect_line_kwargs = _validate_style_kwargs(default_line_kwargs, line_kwargs)

        if kind == "actual_vs_predicted":
            self.line_ = ax.plot(
                x_range_perfect_pred, y_range_perfect_pred, **perfect_line_kwargs
            )[0]
            ax.set(
                aspect="equal",
                xlim=x_range_perfect_pred,
                ylim=y_range_perfect_pred,
                xticks=np.linspace(
                    x_range_perfect_pred[0], x_range_perfect_pred[1], num=5
                ),
                yticks=np.linspace(
                    y_range_perfect_pred[0], y_range_perfect_pred[1], num=5
                ),
            )
        else:
            self.line_ = ax.plot(x_range_perfect_pred, [0, 0], **perfect_line_kwargs)[0]
            ax.set(
                xlim=x_range_perfect_pred,
                ylim=y_range_perfect_pred,
                xticks=np.linspace(
                    x_range_perfect_pred[0], x_range_perfect_pred[1], num=5
                ),
                yticks=np.linspace(
                    y_range_perfect_pred[0], y_range_perfect_pred[1], num=5
                ),
            )

        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.legend(title=estimator_name)

        self.ax_ = ax
        self.figure_ = ax.figure

        if despine:
            x_range = self.ax_.get_xlim()
            y_range = self.ax_.get_ylim()
            _despine_matplotlib_axis(self.ax_, x_range=x_range, y_range=y_range)

    @classmethod
    def _from_predictions(
        cls,
        y_true: list[ArrayLike],
        y_pred: list[ArrayLike],
        *,
        estimator: BaseEstimator,  # currently only for consistency with other plots
        estimator_name: str,
        ml_task: MLTask,  # FIXME: to be used when having single-output vs. multi-output
        data_source: Literal["train", "test", "X_y"],
        subsample: Union[float, int, None] = 1_000,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> "PredictionErrorDisplay":
        """Plot the prediction error given the true and predicted targets.

        Parameters
        ----------
        y_true : list of array-like of shape (n_samples,)
            True target values.

        y_pred : list of array-like of shape (n_samples,)
            Predicted target values.

        estimator : estimator instance
            The estimator from which `y_pred` is obtained.

        estimator_name : str,
            The name of the estimator.

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
        rng: np.random.RandomState = check_random_state(random_state)
        if isinstance(subsample, numbers.Integral):
            if subsample <= 0:
                raise ValueError(
                    f"When an integer, subsample={subsample} should be positive."
                )
        elif isinstance(subsample, numbers.Real) and (subsample <= 0 or subsample >= 1):
            raise ValueError(
                f"When a floating-point, subsample={subsample} should be in the "
                "(0, 1) range."
            )

        y_true_display, y_pred_display = [], []
        for y_true_i, y_pred_i in zip(y_true, y_pred):
            n_samples = _num_samples(y_true_i)
            if subsample is None:
                subsample_ = n_samples
            elif isinstance(subsample, numbers.Integral):
                subsample_ = subsample
            else:  # subsample is a float
                subsample_ = int(n_samples * subsample)

            # normalize subsample based on the number of splits
            subsample_ = int(subsample_ / len(y_true))
            if subsample_ < n_samples:
                indices = rng.choice(np.arange(n_samples), size=subsample_)
                y_true_display.append(
                    check_array(
                        _safe_indexing(y_true_i, indices, axis=0), ensure_2d=False
                    )
                )
                y_pred_display.append(
                    check_array(
                        _safe_indexing(y_pred_i, indices, axis=0), ensure_2d=False
                    )
                )
            else:
                y_true_display.append(y_true_i)
                y_pred_display.append(y_pred_i)

        return cls(
            y_true=y_true_display,
            y_pred=y_pred_display,
            estimator_name=estimator_name,
            data_source=data_source,
        )
