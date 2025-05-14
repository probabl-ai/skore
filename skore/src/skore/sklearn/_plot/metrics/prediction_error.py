import numbers
from collections import namedtuple
from typing import Any, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from numpy.typing import NDArray
from sklearn.utils.validation import _num_samples, check_array

from skore.externals._sklearn_compat import _safe_indexing
from skore.sklearn._plot.style import StyleDisplayMixin
from skore.sklearn._plot.utils import (
    HelpDisplayMixin,
    _despine_matplotlib_axis,
    _validate_style_kwargs,
    sample_mpl_colormap,
)
from skore.sklearn.types import MLTask, YPlotData

RangeData = namedtuple("RangeData", ["min", "max"])


class PredictionErrorDisplay(StyleDisplayMixin, HelpDisplayMixin):
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

    residuals : list of ndarray of shape (n_samples,)
        Residuals. Equal to `y_true - y_pred`.

    range_y_true : RangeData
        Global range of the true values.

    range_y_pred : RangeData
        Global range of the predicted values.

    range_residuals : RangeData
        Global range of the residuals.

    estimator_names : list of str
        Name of the estimators.

    data_source : {"train", "test", "X_y"}
        The data source used to display the prediction error.

    ml_task : {"regression", "multioutput-regression"}
        The machine learning task.

    report_type : {"cross-validation", "estimator", "comparison-estimator"}
        The type of report.

    Attributes
    ----------
    line_ : matplotlib Artist
        Optimal line representing `y_true == y_pred`. Therefore, it is a
        diagonal line for `kind="predictions"` and a horizontal line for
        `kind="residuals"`.

    errors_lines_ : matplotlib Artist or None
        Residual lines. If `with_errors=False`, then it is set to `None`.

    scatter_ : list of matplotlib Artist
        Scatter data points.

    ax_ : matplotlib Axes
        Axes with the different matplotlib axis.

    figure_ : matplotlib Figure
        Figure containing the scatter and lines.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.linear_model import Ridge
    >>> from skore import train_test_split
    >>> from skore import EstimatorReport
    >>> X, y = load_diabetes(return_X_y=True)
    >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
    >>> classifier = Ridge()
    >>> report = EstimatorReport(classifier, **split_data)
    >>> display = report.metrics.prediction_error()
    >>> display.plot(kind="actual_vs_predicted")
    """

    _default_data_points_kwargs: Union[dict[str, Any], None] = None
    _default_perfect_model_kwargs: Union[dict[str, Any], None] = None

    def __init__(
        self,
        *,
        y_true: list[NDArray],
        y_pred: list[NDArray],
        residuals: list[NDArray],
        range_y_true: RangeData,
        range_y_pred: RangeData,
        range_residuals: RangeData,
        estimator_names: list[str],
        data_source: Literal["train", "test", "X_y"],
        ml_task: MLTask,
        report_type: Literal["cross-validation", "estimator", "comparison-estimator"],
    ) -> None:
        self.y_true = y_true
        self.y_pred = y_pred
        self.residuals = residuals
        self.range_y_true = range_y_true
        self.range_y_pred = range_y_pred
        self.range_residuals = range_residuals
        self.estimator_names = estimator_names
        self.data_source = data_source
        self.ml_task = ml_task
        self.report_type = report_type

    def _validate_data_points_kwargs(
        self,
        *,
        data_points_kwargs: Union[dict[str, Any], list[dict[str, Any]], None],
    ) -> list[dict[str, Any]]:
        """Validate and format the scatter keyword arguments.

        Parameters
        ----------
        data_points_kwargs : dict or list of dict or None
            Keyword arguments for the scatter plot.

        Returns
        -------
        list of dict
            Validated list of keyword arguments for each curve.

        Raises
        ------
        ValueError
            If the format of `data_points_kwargs` is invalid.
        """
        if data_points_kwargs is None:
            return [{}] * len(self.y_true)
        elif len(self.y_true) == 1:
            if isinstance(data_points_kwargs, dict):
                return [data_points_kwargs]
            raise ValueError(
                "You intend to plot the prediction error for a single estimator. We "
                "expect `data_points_kwargs` to be a dictionary. Got "
                f"{type(data_points_kwargs)} instead."
            )
        elif not isinstance(data_points_kwargs, list) or len(data_points_kwargs) != len(
            self.y_true
        ):
            raise ValueError(
                "You intend to plot prediction errors either from multiple estimators "
                "or from a cross-validated estimator. We expect `data_points_kwargs` "
                "to be a list of dictionaries with the same length as the number of "
                "estimators or splits. Got "
                f"{len(data_points_kwargs)} instead of {len(self.y_true)}."
            )

        return data_points_kwargs

    def _plot_single_estimator(
        self,
        *,
        kind: Literal["actual_vs_predicted", "residual_vs_predicted"],
        estimator_name: str,
        samples_kwargs: list[dict[str, Any]],
    ) -> list[Artist]:
        """Plot the prediction error for a single estimator.

        Parameters
        ----------
        kind : {"actual_vs_predicted", "residual_vs_predicted"}
            The type of plot to draw.

        estimator_name : str
            Name of the estimator.

        samples_kwargs : list of dict
            Keyword arguments for the scatter plot.

        Returns
        -------
        scatter : list of matplotlib Artist
            The scatter plot.
        """
        scatter = []
        data_points_kwargs: dict[str, Any] = {
            "color": "tab:blue",
            "alpha": 0.3,
            "s": 10,
        }

        data_points_kwargs_validated = _validate_style_kwargs(
            data_points_kwargs, samples_kwargs[0]
        )

        y_true, y_pred, residuals = self.y_true[0], self.y_pred[0], self.residuals[0]
        if self.data_source in ("train", "test"):
            scatter_label = f"{self.data_source.title()} set"
        else:  # data_source == "X_y"
            scatter_label = "Data set"

        if kind == "actual_vs_predicted":
            scatter.append(
                self.ax_.scatter(
                    y_pred,
                    y_true,
                    label=scatter_label,
                    **data_points_kwargs_validated,
                )
            )
        else:  # kind == "residual_vs_predicted"
            scatter.append(
                self.ax_.scatter(
                    y_pred,
                    residuals,
                    label=scatter_label,
                    **data_points_kwargs_validated,
                )
            )

        self.ax_.legend(bbox_to_anchor=(1.02, 1), title=estimator_name)

        return scatter

    def _plot_cross_validated_estimator(
        self,
        *,
        kind: Literal["actual_vs_predicted", "residual_vs_predicted"],
        estimator_name: str,
        samples_kwargs: list[dict[str, Any]],
    ) -> list[Artist]:
        """Plot the prediction error for a cross-validated estimator.

        Parameters
        ----------
        kind : {"actual_vs_predicted", "residual_vs_predicted"}
            The type of plot to draw.

        estimator_name : str
            Name of the estimator.

        samples_kwargs : list of dict
            Keyword arguments for the scatter plot.

        Returns
        -------
        scatter : list of matplotlib Artist
            The scatter plot.
        """
        scatter = []
        data_points_kwargs: dict[str, Any] = {"alpha": 0.3, "s": 10}
        colors_markers = sample_mpl_colormap(
            colormaps.get_cmap("tab10"),
            len(self.y_true) if len(self.y_true) > 10 else 10,
        )

        for split_idx in range(len(self.y_true)):
            data_points_kwargs_fold = {
                "color": colors_markers[split_idx],
                **data_points_kwargs,
            }

            data_points_kwargs_validated = _validate_style_kwargs(
                data_points_kwargs_fold, samples_kwargs[split_idx]
            )

            label = f"Estimator of fold #{split_idx + 1}"

            if kind == "actual_vs_predicted":
                scatter.append(
                    self.ax_.scatter(
                        self.y_pred[split_idx],
                        self.y_true[split_idx],
                        label=label,
                        **data_points_kwargs_validated,
                    )
                )
            else:  # kind == "residual_vs_predicted"
                scatter.append(
                    self.ax_.scatter(
                        self.y_pred[split_idx],
                        self.residuals[split_idx],
                        label=label,
                        **data_points_kwargs_validated,
                    )
                )

        if self.data_source in ("train", "test"):
            title = f"{estimator_name} on $\\bf{{{self.data_source}}}$ set"
        else:
            title = f"{estimator_name} on $\\bf{{external}}$ set"
        self.ax_.legend(bbox_to_anchor=(1.02, 1), title=title)

        return scatter

    def _plot_comparison_estimator(
        self,
        *,
        kind: Literal["actual_vs_predicted", "residual_vs_predicted"],
        estimator_names: list[str],
        samples_kwargs: list[dict[str, Any]],
    ) -> list[Artist]:
        """Plot the prediction error of several estimators.

        Parameters
        ----------
        kind : {"actual_vs_predicted", "residual_vs_predicted"}
            The type of plot to draw.

        estimator_names : list of str
            Name of the estimators.

        samples_kwargs : list of dict
            Keyword arguments for the scatter plot.

        Returns
        -------
        scatter : list of matplotlib Artist
            The scatter plot.
        """
        scatter = []
        data_points_kwargs: dict[str, Any] = {"alpha": 0.3, "s": 10}
        colors_markers = sample_mpl_colormap(
            colormaps.get_cmap("tab10"),
            len(self.y_true) if len(self.y_true) > 10 else 10,
        )

        for estimator_idx in range(len(self.y_true)):
            data_points_kwargs_fold = {
                "color": colors_markers[estimator_idx],
                **data_points_kwargs,
            }

            data_points_kwargs_validated = _validate_style_kwargs(
                data_points_kwargs_fold, samples_kwargs[estimator_idx]
            )

            label = f"{estimator_names[estimator_idx]}"

            if kind == "actual_vs_predicted":
                scatter.append(
                    self.ax_.scatter(
                        self.y_pred[estimator_idx],
                        self.y_true[estimator_idx],
                        label=label,
                        **data_points_kwargs_validated,
                    )
                )
            else:  # kind == "residual_vs_predicted"
                scatter.append(
                    self.ax_.scatter(
                        self.y_pred[estimator_idx],
                        self.residuals[estimator_idx],
                        label=label,
                        **data_points_kwargs_validated,
                    )
                )

        self.ax_.legend(
            bbox_to_anchor=(1.02, 1),
            title=f"Prediction errors on $\\bf{{{self.data_source}}}$ set",
        )

        return scatter

    @StyleDisplayMixin.style_plot
    def plot(
        self,
        ax: Optional[Axes] = None,
        *,
        estimator_name: Optional[str] = None,
        kind: Literal[
            "actual_vs_predicted", "residual_vs_predicted"
        ] = "residual_vs_predicted",
        data_points_kwargs: Optional[
            Union[dict[str, Any], list[dict[str, Any]]]
        ] = None,
        perfect_model_kwargs: Optional[dict[str, Any]] = None,
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

        data_points_kwargs : dict, default=None
            Dictionary with keywords passed to the `matplotlib.pyplot.scatter`
            call.

        perfect_model_kwargs : dict, default=None
            Dictionary with keyword passed to the `matplotlib.pyplot.plot`
            call to draw the optimal line.

        despine : bool, default=True
            Whether to remove the top and right spines from the plot.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> classifier = Ridge()
        >>> report = EstimatorReport(classifier, **split_data)
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

        self.figure_, self.ax_ = (ax.figure, ax) if ax is not None else plt.subplots()

        perfect_model_kwargs_validated = _validate_style_kwargs(
            {
                "color": "black",
                "alpha": 0.7,
                "linestyle": "--",
                "label": "Perfect predictions",
            },
            perfect_model_kwargs or self._default_perfect_model_kwargs or {},
        )

        if kind == "actual_vs_predicted":
            # For actual vs predicted, we want the same range for both axes
            min_value = min(self.range_y_pred.min, self.range_y_true.min)
            max_value = max(self.range_y_pred.max, self.range_y_true.max)
            x_range_perfect_pred = [min_value, max_value]
            y_range_perfect_pred = [min_value, max_value]

            self.line_ = self.ax_.plot(
                x_range_perfect_pred,
                y_range_perfect_pred,
                **perfect_model_kwargs_validated,
            )[0]
            self.ax_.set(
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

        else:  # kind == "residual_vs_predicted"
            x_range_perfect_pred = [self.range_y_pred.min, self.range_y_pred.max]
            y_range_perfect_pred = [self.range_residuals.min, self.range_residuals.max]

            self.line_ = self.ax_.plot(
                x_range_perfect_pred, [0, 0], **perfect_model_kwargs_validated
            )[0]
            self.ax_.set(
                xlim=x_range_perfect_pred,
                ylim=y_range_perfect_pred,
                xticks=np.linspace(
                    x_range_perfect_pred[0], x_range_perfect_pred[1], num=5
                ),
                yticks=np.linspace(
                    y_range_perfect_pred[0], y_range_perfect_pred[1], num=5
                ),
            )

        self.ax_.set(xlabel=xlabel, ylabel=ylabel)

        # make the scatter plot afterwards since it should take into account the line
        # for the perfect predictions
        if data_points_kwargs is None:
            data_points_kwargs = self._default_data_points_kwargs
        data_points_kwargs = self._validate_data_points_kwargs(
            data_points_kwargs=data_points_kwargs
        )

        if self.report_type == "estimator":
            self.scatter_ = self._plot_single_estimator(
                kind=kind,
                estimator_name=(
                    self.estimator_names[0]
                    if estimator_name is None
                    else estimator_name
                ),
                samples_kwargs=data_points_kwargs,
            )
        elif self.report_type == "cross-validation":
            self.scatter_ = self._plot_cross_validated_estimator(
                kind=kind,
                estimator_name=(
                    self.estimator_names[0]
                    if estimator_name is None
                    else estimator_name
                ),
                samples_kwargs=data_points_kwargs,
            )
        elif self.report_type == "comparison-estimator":
            self.scatter_ = self._plot_comparison_estimator(
                kind=kind,
                estimator_names=self.estimator_names,
                samples_kwargs=data_points_kwargs,
            )
        else:
            raise ValueError(
                f"`report_type` should be one of 'estimator', 'cross-validation', "
                f"or 'comparison-estimator'. Got '{self.report_type}' instead."
            )

        if despine:
            x_range = self.ax_.get_xlim()
            y_range = self.ax_.get_ylim()
            _despine_matplotlib_axis(self.ax_, x_range=x_range, y_range=y_range)

    @classmethod
    def _compute_data_for_display(
        cls,
        y_true: list[YPlotData],
        y_pred: list[YPlotData],
        *,
        report_type: Literal["cross-validation", "estimator", "comparison-estimator"],
        estimator_names: list[str],
        ml_task: MLTask,
        data_source: Literal["train", "test", "X_y"],
        subsample: Union[float, int, None] = 1_000,
        seed: Optional[int] = None,
        **kwargs,
    ) -> "PredictionErrorDisplay":
        """Plot the prediction error given the true and predicted targets.

        Parameters
        ----------
        y_true : list of array-like of shape (n_samples,)
            True target values.

        y_pred : list of array-like of shape (n_samples,)
            Predicted target values.

        estimators : list of estimator instances
            The estimators from which `y_pred` is obtained.

        ml_task : {"regression", "multioutput-regression"}
            The machine learning task.

        data_source : {"train", "test", "X_y"}
            The data source used to compute the prediction error curve.

        subsample : float, int or None, default=1_000
            Sampling the samples to be shown on the scatter plot. If `float`,
            it should be between 0 and 1 and represents the proportion of the
            original dataset. If `int`, it represents the number of samples
            display on the scatter plot. If `None`, no subsampling will be
            applied. by default, 1000 samples or less will be displayed.

        seed : int, default=None
            The seed used to initialize the random number generator used for the
            subsampling.

        **kwargs : dict
            Additional keyword arguments that are ignored for compatibility with
            other metrics displays. Here, `estimators` is ignored.

        Returns
        -------
        display : PredictionErrorDisplay
        """
        rng = np.random.default_rng(seed)
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

        if ml_task != "regression":  # pragma: no cover
            raise ValueError(
                "The machine learning task must be 'regression'. "
                f"Got {ml_task} instead."
            )

        y_true_display, y_pred_display, residuals_display = [], [], []
        y_true_min, y_true_max = np.inf, -np.inf
        y_pred_min, y_pred_max = np.inf, -np.inf
        residuals_min, residuals_max = np.inf, -np.inf

        for y_true_i, y_pred_i in zip(y_true, y_pred):
            n_samples = _num_samples(y_true_i.y)
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
                y_true_sample = check_array(
                    _safe_indexing(y_true_i.y, indices, axis=0), ensure_2d=False
                )
                y_pred_sample = check_array(
                    _safe_indexing(y_pred_i.y, indices, axis=0), ensure_2d=False
                )
                residuals_sample = y_true_sample - y_pred_sample

                y_true_display.append(y_true_sample)
                y_pred_display.append(y_pred_sample)
                residuals_display.append(residuals_sample)
            else:
                y_true_sample = y_true_i.y
                y_pred_sample = y_pred_i.y
                residuals_sample = y_true_i.y - y_pred_i.y

                y_true_display.append(y_true_sample)
                y_pred_display.append(y_pred_sample)
                residuals_display.append(residuals_sample)

            y_true_min = min(y_true_min, np.min(y_true_sample))
            y_true_max = max(y_true_max, np.max(y_true_sample))
            y_pred_min = min(y_pred_min, np.min(y_pred_sample))
            y_pred_max = max(y_pred_max, np.max(y_pred_sample))
            residuals_min = min(residuals_min, np.min(residuals_sample))
            residuals_max = max(residuals_max, np.max(residuals_sample))

        range_y_true = RangeData(min=y_true_min, max=y_true_max)
        range_y_pred = RangeData(min=y_pred_min, max=y_pred_max)
        range_residuals = RangeData(min=residuals_min, max=residuals_max)

        return cls(
            y_true=y_true_display,
            y_pred=y_pred_display,
            residuals=residuals_display,
            range_y_true=range_y_true,
            range_y_pred=range_y_pred,
            range_residuals=range_residuals,
            estimator_names=estimator_names,
            data_source=data_source,
            ml_task=ml_task,
            report_type=report_type,
        )
