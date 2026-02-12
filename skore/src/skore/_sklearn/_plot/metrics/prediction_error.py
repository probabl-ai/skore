import numbers
from collections import namedtuple
from typing import Literal, cast

import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from pandas import DataFrame
from sklearn.utils.validation import _num_samples, check_array

from skore._externals._sklearn_compat import _safe_indexing
from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn._plot.utils import (
    _despine_matplotlib_axis,
    _validate_style_kwargs,
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

        - `estimator`
        - `split` (may be null)
        - `y_true`
        - `y_pred`
        - `residuals`
        - `output`.

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
    facet_ : seaborn FacetGrid
        FacetGrid containing the prediction error.

    figure_ : matplotlib Figure
        The figure on which the prediction error is plotted.

    ax_ : matplotlib Axes
        The axes on which the prediction error is plotted.

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

    _default_relplot_kwargs = {
        "alpha": 0.5,
        "s": 15,
        "marker": "o",
        "aspect": 1.0,
    }
    _default_perfect_model_kwargs = {
        "color": "black",
        "alpha": 0.7,
        "linestyle": "--",
    }

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

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        subplot_by: Literal["auto", "data_source", "split", "estimator", "output"]
        | None = "auto",
        kind: Literal[
            "actual_vs_predicted", "residual_vs_predicted"
        ] = "residual_vs_predicted",
        despine: bool = True,
    ) -> None:
        """Plot visualization.

        Extra keyword arguments will be passed to matplotlib's ``plot``.

        Parameters
        ----------
        subplot_by : {"auto", "data_source", "split", "estimator", "output", \
                None}, default="auto"
            The variable to use for creating subplots:

            - "auto" creates subplots by estimator for comparison reports, otherwise
              uses a single plot.
            - "data_source" creates subplots by data source (train/test).
            - "split" creates subplots by cross-validation split.
            - "estimator" creates subplots by estimator.
            - "output" creates subplots by output target
            - None creates a single plot.

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
            subplot_by=subplot_by,
            kind=kind,
            despine=despine,
        )

    def _plot_matplotlib(
        self,
        *,
        subplot_by: Literal["auto", "data_source", "split", "estimator", "output"]
        | None = "auto",
        kind: Literal[
            "actual_vs_predicted", "residual_vs_predicted"
        ] = "residual_vs_predicted",
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
            y_plot = "y_true"
            min_value = min(self.range_y_pred.min, self.range_y_true.min)
            max_value = max(self.range_y_pred.max, self.range_y_true.max)
            x_range_perfect_pred = [min_value - 0.1, max_value + 0.1]
            y_range_perfect_pred = [min_value - 0.1, max_value + 0.1]
            y_line = y_range_perfect_pred
        else:  # kind == "residual_vs_predicted"
            xlabel, ylabel = "Predicted values", "Residuals (actual - predicted)"
            y_plot = "residuals"
            x_range_perfect_pred = [
                self.range_y_pred.min - 0.1,
                self.range_y_pred.max + 0.1,
            ]
            y_range_perfect_pred = [
                self.range_residuals.min - 0.1,
                self.range_residuals.max + 0.1,
            ]
            y_line = [0, 0]

        plot_data = self.frame()
        col, hue, style = self._get_plot_columns(subplot_by)
        relplot_kwargs = {
            "col": col,
            "hue": hue,
            "style": style,
            **self._default_relplot_kwargs,
        }
        if hue:
            relplot_kwargs["hue_order"] = plot_data[hue].unique().tolist()
        if style == "data_source":
            relplot_kwargs["style_order"] = ["train", "test"]
            if hue == "data_source":
                relplot_kwargs["hue_order"] = ["train", "test"]

        self.facet_ = sns.relplot(
            data=plot_data,
            x="y_pred",
            y=y_plot,
            kind="scatter",
            **_validate_style_kwargs(relplot_kwargs, {}),
        )
        self.figure_, self.ax_ = self.facet_.figure, self.facet_.axes.flatten()

        title = "Prediction Error"
        if "comparison" not in self.report_type:
            title += f" for {self._prediction_error['estimator'].cat.categories.item()}"
        title += (
            f"\nData source: {self.data_source.capitalize()} set"
            if self.data_source in ("train", "test")
            else "\nData source: external set"
            if self.data_source == "X_y"
            else "\nData source: Train and Test"
        )
        self.figure_.suptitle(title)

        for ax in self.ax_:
            ax.plot(
                x_range_perfect_pred,
                y_line,
                **self._default_perfect_model_kwargs,
            )
            ax.set(
                xlabel=xlabel,
                ylabel=ylabel,
                xlim=x_range_perfect_pred,
                ylim=y_range_perfect_pred,
                xticks=np.linspace(
                    x_range_perfect_pred[0], x_range_perfect_pred[1], num=5
                ),
                yticks=np.linspace(
                    y_range_perfect_pred[0], y_range_perfect_pred[1], num=5
                ),
            )

            if despine:
                _despine_matplotlib_axis(
                    ax, x_range=ax.get_xlim(), y_range=ax.get_ylim()
                )

        # Add the perfect model line to the legend
        # We retrieve the legend elements created by seaborn, add the perfect model line
        # and create a new legend manually.
        handles = []
        labels = []
        if self.facet_._legend is not None:
            handles = list(self.facet_._legend.legend_handles)
            labels = [t.get_text() for t in self.facet_._legend.get_texts()]
            self.facet_._legend.remove()
            if hue == "split":
                labels = [f"Split #{label}" for label in labels]
            if hue == "output" and style is None:
                labels = [f"Output #{label}" for label in labels]
        handles.append(
            Line2D([0], [0], **self._default_perfect_model_kwargs)  # type: ignore[arg-type]
        )

        labels.append("Perfect predictions")

        self.figure_.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            ncol=1,
            frameon=True,
        )

        if len(self.ax_) == 1:
            self.ax_ = self.ax_[0]

    def _get_plot_columns(
        self,
        subplot_by: Literal["auto", "estimator", "data_source", "split", "output"]
        | None,
    ) -> tuple[str | None, str | None, str | None]:
        """Validate the `subplot_by` parameter.

        Parameters
        ----------
        subplot_by : {"auto", "estimator", "data_source", "split", "output", \
                None}
            The variable to use for subplotting.

        Returns
        -------
        tuple of (str or None, str or None, str or None)
            A tuple containing (col, hue, style) where:
            - col: Column variable for subplots
            - hue: Variable for color encoding
            - style: Variable for marker style
        """
        valid_subplot_by: list[str | None] = ["auto"]
        hue_candidates = []

        if self.data_source == "both" and (
            self.ml_task != "multioutput-regression"
            or "comparison" not in self.report_type
        ):
            valid_subplot_by.append("data_source")
        if self.ml_task == "multioutput-regression":
            valid_subplot_by.append("output")
            hue_candidates.append("output")
        if "comparison" in self.report_type:
            valid_subplot_by.append("estimator")
            hue_candidates.append("estimator")
        else:
            valid_subplot_by.append(None)
        if "cross-validation" in self.report_type and (
            self.ml_task != "multioutput-regression"
            or "comparison" not in self.report_type
        ):
            valid_subplot_by.append("split")
            hue_candidates.append("split")

        if subplot_by not in valid_subplot_by:
            raise ValueError(
                f"Invalid `subplot_by` parameter. Valid options are: "
                f"{', '.join(str(s) for s in valid_subplot_by)}. "
                f"Got '{subplot_by}' instead."
            )

        if subplot_by == "auto":
            col = "estimator" if "comparison" in self.report_type else None
        else:
            col = subplot_by
        hue = hue[0] if (hue := [c for c in hue_candidates if c != col]) else None
        style = (
            "data_source"
            if self.data_source == "both" and col != "data_source"
            else None
        )

        return col, hue, style

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

        if ml_task not in ["regression", "multioutput-regression"]:  # pragma: no cover
            raise ValueError(
                "The machine learning task must be 'regression' or"
                f" 'multioutput-regression'. Got {ml_task} instead."
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
            else:
                y_true_sample = cast(np.typing.NDArray, y_true_i.y)
                y_pred_sample = cast(np.typing.NDArray, y_pred_i.y)

            residuals_sample = y_true_sample - y_pred_sample
            if ml_task == "multioutput-regression":
                for output in range(y_true_sample.shape[1]):
                    for y_true_sample_i, y_pred_sample_i, residuals_sample_i in zip(
                        y_true_sample[:, output],
                        y_pred_sample[:, output],
                        residuals_sample[:, output],
                        strict=True,
                    ):
                        prediction_error_records.append(
                            {
                                "estimator": y_true_i.estimator_name,
                                "data_source": y_true_i.data_source,
                                "split": y_true_i.split,
                                "y_true": y_true_sample_i,
                                "y_pred": y_pred_sample_i,
                                "residuals": residuals_sample_i,
                                "output": output,
                            }
                        )
            else:
                for y_true_sample_i, y_pred_sample_i, residuals_sample_i in zip(
                    y_true_sample, y_pred_sample, residuals_sample, strict=True
                ):
                    prediction_error_records.append(
                        {
                            "estimator": y_true_i.estimator_name,
                            "data_source": y_true_i.data_source,
                            "split": y_true_i.split,
                            "y_true": y_true_sample_i,
                            "y_pred": y_pred_sample_i,
                            "residuals": residuals_sample_i,
                            "output": np.nan,
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

        dtypes = {
            "estimator": "category",
            "data_source": "category",
            "split": "category",
        }
        if ml_task == "multioutput-regression":
            dtypes["output"] = "category"

        return cls(
            prediction_error=DataFrame.from_records(prediction_error_records).astype(
                dtypes
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

            - `estimator`: Name of the estimator (when comparing estimators)
            - `split`: Cross-validation split ID (when doing cross-validation)
            - `y_true`: True target values
            - `y_pred`: Predicted target values
            - `residuals`: Difference between true and predicted values
              `(y_true - y_pred)`
            - `output`: Index of the output target (for multioutput-regression)


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

        if self.ml_task == "multioutput-regression":
            statistical_columns.append("output")

        if self.report_type == "estimator":
            columns = statistical_columns
        elif self.report_type == "cross-validation":
            columns = ["split"] + statistical_columns
        elif self.report_type == "comparison-estimator":
            columns = ["estimator"] + statistical_columns
        else:  # self.report_type == "comparison-cross-validation"
            columns = ["estimator", "split"] + statistical_columns

        return self._prediction_error[columns]

    # ignore the type signature because we override kwargs by specifying the name of
    # the parameters for the user.
    def set_style(  # type: ignore[override]
        self,
        *,
        policy: Literal["override", "update"] = "update",
        relplot_kwargs: dict | None = None,
        perfect_model_kwargs: dict | None = None,
    ):
        """Set the style parameters for the display.

        Parameters
        ----------
        policy : {"override", "update"}, default="update"
            Policy to use when setting the style parameters.
            If "override", existing settings are set to the provided values.
            If "update", existing settings are not changed; only settings that were
            previously unset are changed.

        relplot_kwargs : dict, default=None
            Additional keyword arguments to be passed to :func:`seaborn.relplot` for
            rendering the scatter plot(s). Common options include `palette`, `alpha`,
            `s`, `marker`, etc.

        perfect_model_kwargs : dict, default=None
            Additional keyword arguments to be passed to :func:`matplotlib.pyplot.plot`
            for drawing the perfect prediction line. Common options include `color`,
            `alpha`, `linestyle`, etc.

        Returns
        -------
        self : object
            The instance with a modified style.

        Raises
        ------
        ValueError
            If a style parameter is unknown.
        """
        return super().set_style(
            policy=policy,
            relplot_kwargs=relplot_kwargs or {},
            perfect_model_kwargs=perfect_model_kwargs or {},
        )
