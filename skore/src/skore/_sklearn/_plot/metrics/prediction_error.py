import numbers
from collections import namedtuple
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.artist import Artist
from pandas import DataFrame
from sklearn.utils.validation import _num_samples, check_array

from skore._externals._sklearn_compat import _safe_indexing
from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn._plot.utils import (
    _despine_matplotlib_axis,
    _validate_style_kwargs,
    sample_mpl_colormap,
)
from skore._sklearn.types import DataSource, MLTask, ReportType, YPlotData

RangeData = namedtuple("RangeData", ["min", "max"])

MAX_N_LABELS = 6  # 5 + 1 for the perfect model line


class PredictionErrorDisplay(DisplayMixin):
    """Visualization of the prediction error of a regression model.

    This tool can display "residuals vs predicted" or "actual vs predicted"
    using scatter plots to qualitatively assess the behavior of a regressor,
    preferably on held-out data points.

    An instance of this class should be created by
    `EstimatorReport.metrics.prediction_error()`.
    You should not create an instance of this class directly.

    Parameters
    ----------
    prediction_error : DataFrame
        The prediction error data to display. The columns are

        - `estimator_name`
        - `split` (may be null)
        - `y_true`
        - `y_pred`
        - `residuals`.

    range_y_true : RangeData
        Global range of the true values.

    range_y_pred : RangeData
        Global range of the predicted values.

    range_residuals : RangeData
        Global range of the residuals.

    data_source : {"train", "test", "X_y", "both"}
        The data source used to display the prediction error.

    ml_task : {"regression", "multioutput-regression"}
        The machine learning task.

    report_type : {"comparison-cross-validation", "comparison-estimator", \
            "cross-validation", "estimator"}
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

    _default_data_points_kwargs: dict[str, Any] | None = None
    _default_perfect_model_kwargs: dict[str, Any] | None = None

    def __init__(
        self,
        *,
        prediction_error: DataFrame,
        range_y_true: RangeData,
        range_y_pred: RangeData,
        range_residuals: RangeData,
        data_source: DataSource | Literal["both"],
        ml_task: MLTask,
        report_type: ReportType,
    ) -> None:
        self._prediction_error = prediction_error
        self.range_y_true = range_y_true
        self.range_y_pred = range_y_pred
        self.range_residuals = range_residuals
        self.data_source = data_source
        self.ml_task = ml_task
        self.report_type = report_type

    def _validate_data_points_kwargs(
        self,
        *,
        data_points_kwargs: dict[str, Any] | list[dict[str, Any]] | None,
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
        if self.report_type == "estimator":
            allow_single_dict = True
            n_scatter_groups = 1
        elif self.report_type == "cross-validation":
            allow_single_dict = True
            n_scatter_groups = len(self._prediction_error["split"].cat.categories)
        elif self.report_type in (
            "comparison-estimator",
            "comparison-cross-validation",
        ):
            # since we compare different estimators, it does not make sense to share
            # a single dictionary for all the estimators.
            allow_single_dict = False
            n_scatter_groups = len(
                self._prediction_error["estimator_name"].cat.categories
            )
        else:
            raise ValueError(
                f"`report_type` should be one of 'estimator', 'cross-validation', "
                "'comparison-cross-validation' or 'comparison-estimator'. "
                f"Got '{self.report_type}' instead."
            )

        if data_points_kwargs is None:
            return [{}] * n_scatter_groups
        elif n_scatter_groups == 1:
            if isinstance(data_points_kwargs, dict):
                return [data_points_kwargs]
            raise ValueError(
                "You intend to plot the prediction error for a single estimator. We "
                "expect `data_points_kwargs` to be a dictionary. Got "
                f"{type(data_points_kwargs)} instead."
            )
        else:
            if not allow_single_dict and isinstance(data_points_kwargs, dict):
                raise ValueError(
                    "You intend to plot multiple curves. We expect "
                    "`data_points_kwargs` to be a list of dictionaries. Got "
                    f"{type(data_points_kwargs)} instead."
                )

            if isinstance(data_points_kwargs, dict):
                return [data_points_kwargs] * n_scatter_groups

            # From this point, we have a list of dictionaries
            if len(data_points_kwargs) != n_scatter_groups:
                raise ValueError(
                    "You intend to plot prediction errors either from multiple "
                    "estimators or from a cross-validated estimator. "
                    "We expect `data_points_kwargs` to be a list of dictionaries "
                    "with the same length as the number of estimators or splits. "
                    f"Got {len(data_points_kwargs)} instead of {n_scatter_groups}."
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
        data_points_kwargs: dict[str, Any] = {"alpha": 0.3, "s": 10}

        data_points_kwargs_validated = _validate_style_kwargs(
            data_points_kwargs, samples_kwargs[0]
        )

        def add_scatter(data_source: DataSource, marker="o"):
            if data_source in ("train", "test"):
                scatter_label = f"{data_source.title()} set"
            else:  # data_source == "X_y"
                scatter_label = "External data set"

            prediction_error = self._prediction_error.query(
                f"data_source == {data_source!r}"
            )

            scatter.append(
                self.ax_.scatter(
                    x=prediction_error["y_pred"],
                    y=(
                        prediction_error["y_true"]
                        if kind == "actual_vs_predicted"
                        else prediction_error["residuals"]
                    ),
                    label=scatter_label,
                    **(data_points_kwargs_validated | {"marker": marker}),
                )
            )

        if self.data_source == "both":
            add_scatter(data_source="test", marker="x")
            add_scatter(data_source="train")
        else:
            add_scatter(data_source=self.data_source)

        # move the perfect model line to the end of the legend
        handles, labels = self.ax_.get_legend_handles_labels()
        handles.append(handles.pop(0))
        labels.append(labels.pop(0))

        self.ax_.legend(handles, labels, loc="lower right")
        self.ax_.set_title(f"Prediction Error for {estimator_name}")

        return cast(list[Artist], scatter)

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
        n_splits = len(self._prediction_error["split"].cat.categories)
        colors_markers = sample_mpl_colormap(
            colormaps.get_cmap("tab10"),
            n_splits if n_splits > 10 else 10,
        )

        for split_idx, prediction_error_split in self._prediction_error.groupby(
            "split", observed=True
        ):
            data_points_kwargs_split = {
                "color": colors_markers[split_idx],
                **data_points_kwargs,
            }

            data_points_kwargs_validated = _validate_style_kwargs(
                data_points_kwargs_split, samples_kwargs[split_idx]
            )

            label = f"Split #{split_idx + 1}"

            if kind == "actual_vs_predicted":
                scatter.append(
                    self.ax_.scatter(
                        prediction_error_split["y_pred"],
                        prediction_error_split["y_true"],
                        label=label,
                        **data_points_kwargs_validated,
                    )
                )
            else:  # kind == "residual_vs_predicted"
                scatter.append(
                    self.ax_.scatter(
                        prediction_error_split["y_pred"],
                        prediction_error_split["residuals"],
                        label=label,
                        **data_points_kwargs_validated,
                    )
                )

        if self.data_source in ("train", "test"):
            legend_title = f"{self.data_source.capitalize()} set"
        else:
            legend_title = "External set"

        # move the perfect model line to the end of the legend
        handles, labels = self.ax_.get_legend_handles_labels()
        handles.append(handles.pop(0))
        labels.append(labels.pop(0))

        if len(labels) > MAX_N_LABELS or kind == "residual_vs_predicted":
            # too many lines to fit legend in the plot
            self.ax_.legend(
                handles, labels, bbox_to_anchor=(1.02, 1), title=legend_title
            )
        else:
            self.ax_.legend(handles, labels, loc="lower right", title=legend_title)
        self.ax_.set_title(f"Prediction Error for {estimator_name}")

        return cast(list[Artist], scatter)

    def _plot_comparison_estimator(
        self,
        *,
        kind: Literal["actual_vs_predicted", "residual_vs_predicted"],
        samples_kwargs: list[dict[str, Any]],
    ) -> list[Artist]:
        """Plot the prediction error of several estimators.

        Parameters
        ----------
        kind : {"actual_vs_predicted", "residual_vs_predicted"}
            The type of plot to draw.

        samples_kwargs : list of dict
            Keyword arguments for the scatter plot.

        Returns
        -------
        scatter : list of matplotlib Artist
            The scatter plot.
        """
        scatter = []
        data_points_kwargs: dict[str, Any] = {"alpha": 0.3, "s": 10}

        estimator_names = self._prediction_error["estimator_name"].cat.categories
        n_estimators = len(estimator_names)
        colors_markers = sample_mpl_colormap(
            colormaps.get_cmap("tab10"),
            n_estimators if n_estimators > 10 else 10,
        )

        for idx, (estimator_name, prediction_error_estimator) in enumerate(
            self._prediction_error.groupby("estimator_name", observed=True)
        ):
            data_points_kwargs_split = {
                "color": colors_markers[idx],
                **data_points_kwargs,
            }

            data_points_kwargs_validated = _validate_style_kwargs(
                data_points_kwargs_split, samples_kwargs[idx]
            )

            if kind == "actual_vs_predicted":
                scatter.append(
                    self.ax_.scatter(
                        prediction_error_estimator["y_pred"],
                        prediction_error_estimator["y_true"],
                        label=estimator_name,
                        **data_points_kwargs_validated,
                    )
                )
            else:  # kind == "residual_vs_predicted"
                scatter.append(
                    self.ax_.scatter(
                        prediction_error_estimator["y_pred"],
                        prediction_error_estimator["residuals"],
                        label=estimator_name,
                        **data_points_kwargs_validated,
                    )
                )

        if self.data_source in ("train", "test"):
            legend_title = f"{self.data_source.capitalize()} set"
        else:
            legend_title = "External set"

        # move the perfect model line to the end of the legend
        handles, labels = self.ax_.get_legend_handles_labels()
        handles.append(handles.pop(0))
        labels.append(labels.pop(0))

        if len(labels) > MAX_N_LABELS or kind == "residual_vs_predicted":
            # too many lines to fit legend in the plot
            self.ax_.legend(
                handles, labels, bbox_to_anchor=(1.02, 1), title=legend_title
            )
        else:
            self.ax_.legend(handles, labels, loc="lower right", title=legend_title)
        self.ax_.set_title("Prediction Error")

        return cast(list[Artist], scatter)

    def _plot_comparison_cross_validation(
        self,
        *,
        kind: Literal["actual_vs_predicted", "residual_vs_predicted"],
        samples_kwargs: list[dict[str, Any]],
    ) -> list[Artist]:
        """Plot the prediction error of several cross-validated estimators.

        Parameters
        ----------
        kind : {"actual_vs_predicted", "residual_vs_predicted"}
            The type of plot to draw.

        samples_kwargs : list of dict
            Keyword arguments for the scatter plot.

        Returns
        -------
        scatter : list of matplotlib Artist
            The scatter plot.
        """
        scatter = []
        data_points_kwargs: dict[str, Any] = {"alpha": 0.3, "s": 10}

        estimator_names = self._prediction_error["estimator_name"].cat.categories
        n_estimators = len(estimator_names)
        colors_markers = sample_mpl_colormap(
            colormaps.get_cmap("tab10"),
            n_estimators if n_estimators > 10 else 10,
        )

        for idx, (estimator_name, prediction_error_estimator) in enumerate(
            self._prediction_error.groupby("estimator_name", observed=True)
        ):
            data_points_kwargs_split = {
                "color": colors_markers[idx],
                **data_points_kwargs,
            }

            data_points_kwargs_validated = _validate_style_kwargs(
                data_points_kwargs_split, samples_kwargs[idx]
            )

            if kind == "actual_vs_predicted":
                scatter.append(
                    self.ax_.scatter(
                        prediction_error_estimator["y_pred"],
                        prediction_error_estimator["y_true"],
                        label=estimator_name,
                        **data_points_kwargs_validated,
                    )
                )
            else:  # kind == "residual_vs_predicted"
                scatter.append(
                    self.ax_.scatter(
                        prediction_error_estimator["y_pred"],
                        prediction_error_estimator["residuals"],
                        label=estimator_name,
                        **data_points_kwargs_validated,
                    )
                )

        if self.data_source in ("train", "test"):
            legend_title = f"{self.data_source.capitalize()} set"
        else:
            legend_title = "External set"

        # move the perfect model line to the end of the legend
        handles, labels = self.ax_.get_legend_handles_labels()
        handles.append(handles.pop(0))
        labels.append(labels.pop(0))

        if len(labels) > MAX_N_LABELS or kind == "residual_vs_predicted":
            # too many lines to fit legend in the plot
            self.ax_.legend(
                handles, labels, bbox_to_anchor=(1.02, 1), title=legend_title
            )
        else:
            self.ax_.legend(handles, labels, loc="lower right", title=legend_title)
        self.ax_.set_title("Prediction Error")

        return cast(list[Artist], scatter)

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        estimator_name: str | None = None,
        kind: Literal[
            "actual_vs_predicted", "residual_vs_predicted"
        ] = "residual_vs_predicted",
        data_points_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        perfect_model_kwargs: dict[str, Any] | None = None,
        despine: bool = True,
    ) -> None:
        """Plot visualization.

        Extra keyword arguments will be passed to matplotlib's ``plot``.

        Parameters
        ----------
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
        return self._plot(
            estimator_name=estimator_name,
            kind=kind,
            data_points_kwargs=data_points_kwargs,
            perfect_model_kwargs=perfect_model_kwargs,
            despine=despine,
        )

    def _plot_matplotlib(
        self,
        *,
        estimator_name: str | None = None,
        kind: Literal[
            "actual_vs_predicted", "residual_vs_predicted"
        ] = "residual_vs_predicted",
        data_points_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        perfect_model_kwargs: dict[str, Any] | None = None,
        despine: bool = True,
    ) -> None:
        """Matplolib implementation of the `plot` method."""
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

        self.figure_, self.ax_ = plt.subplots()

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
                    self._prediction_error["estimator_name"].cat.categories.item()
                    if estimator_name is None
                    else estimator_name
                ),
                samples_kwargs=data_points_kwargs,
            )
        elif self.report_type == "cross-validation":
            self.scatter_ = self._plot_cross_validated_estimator(
                kind=kind,
                estimator_name=(
                    self._prediction_error["estimator_name"].cat.categories.item()
                    if estimator_name is None
                    else estimator_name
                ),
                samples_kwargs=data_points_kwargs,
            )
        elif self.report_type == "comparison-estimator":
            self.scatter_ = self._plot_comparison_estimator(
                kind=kind,
                samples_kwargs=data_points_kwargs,
            )
        elif self.report_type == "comparison-cross-validation":
            self.scatter_ = self._plot_comparison_cross_validation(
                kind=kind,
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
        report_type: ReportType,
        ml_task: MLTask,
        data_source: DataSource | Literal["both"],
        subsample: float | int | None = 1_000,
        seed: int | None = None,
        **kwargs,
    ) -> "PredictionErrorDisplay":
        """Plot the prediction error given the true and predicted targets.

        Parameters
        ----------
        y_true : list of array-like of shape (n_samples,)
            True target values.

        y_pred : list of array-like of shape (n_samples,)
            Predicted target values.

        report_type : {"comparison-cross-validation", "comparison-estimator", \
                "cross-validation", "estimator"}
            The type of report.

        estimators : list of estimator instances
            The estimators from which `y_pred` is obtained.

        ml_task : {"regression", "multioutput-regression"}
            The machine learning task.

        data_source : {"train", "test", "X_y", "both"}
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

        prediction_error_records = []
        y_true_min, y_true_max = np.inf, -np.inf
        y_pred_min, y_pred_max = np.inf, -np.inf
        residuals_min, residuals_max = np.inf, -np.inf

        for y_true_i, y_pred_i in zip(y_true, y_pred, strict=False):
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

                for y_true_sample_i, y_pred_sample_i, residuals_sample_i in zip(
                    y_true_sample, y_pred_sample, residuals_sample, strict=False
                ):
                    prediction_error_records.append(
                        {
                            "estimator_name": y_true_i.estimator_name,
                            "data_source": y_true_i.data_source,
                            "split": y_true_i.split,
                            "y_true": y_true_sample_i,
                            "y_pred": y_pred_sample_i,
                            "residuals": residuals_sample_i,
                        }
                    )
            else:
                y_true_sample = cast(np.typing.NDArray, y_true_i.y)
                y_pred_sample = cast(np.typing.NDArray, y_pred_i.y)
                residuals_sample = y_true_sample - y_pred_sample

                for y_true_sample_i, y_pred_sample_i, residuals_sample_i in zip(
                    y_true_sample, y_pred_sample, residuals_sample, strict=False
                ):
                    prediction_error_records.append(
                        {
                            "estimator_name": y_true_i.estimator_name,
                            "data_source": y_true_i.data_source,
                            "split": y_true_i.split,
                            "y_true": y_true_sample_i,
                            "y_pred": y_pred_sample_i,
                            "residuals": residuals_sample_i,
                        }
                    )

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
            prediction_error=DataFrame.from_records(prediction_error_records).astype(
                {
                    "estimator_name": "category",
                    "data_source": "category",
                    "split": "category",
                }
            ),
            range_y_true=range_y_true,
            range_y_pred=range_y_pred,
            range_residuals=range_residuals,
            data_source=data_source,
            ml_task=ml_task,
            report_type=report_type,
        )

    def frame(self) -> DataFrame:
        """Get the data used to create the prediction error plot.

        Returns
        -------
        DataFrame
            A DataFrame containing the prediction error data with columns depending on
            the report type:

            - `estimator_name`: Name of the estimator (when comparing estimators)
            - `split`: Cross-validation split ID (when doing cross-validation)
            - `y_true`: True target values
            - `y_pred`: Predicted target values
            - `residuals`: Difference between true and predicted values
              `(y_true - y_pred)`


        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import train_test_split, EstimatorReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> reg = Ridge()
        >>> report = EstimatorReport(reg, **split_data)
        >>> display = report.metrics.prediction_error()
        >>> df = display.frame()
        """
        statistical_columns = ["y_true", "y_pred", "residuals"]

        if self.data_source == "both":
            statistical_columns = ["data_source"] + statistical_columns

        if self.report_type == "estimator":
            columns = statistical_columns
        elif self.report_type == "cross-validation":
            columns = ["split"] + statistical_columns
        elif self.report_type == "comparison-estimator":
            columns = ["estimator_name"] + statistical_columns
        else:  # self.report_type == "comparison-cross-validation"
            columns = ["estimator_name", "split"] + statistical_columns

        return self._prediction_error[columns]
