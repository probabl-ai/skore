from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, is_classifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from skore._sklearn._plot.base import BOXPLOT_STYLE, DisplayMixin
from skore._sklearn._plot.utils import _despine_matplotlib_axis
from skore._sklearn.feature_names import _get_feature_names
from skore._sklearn.types import ReportType


class CoefficientsDisplay(DisplayMixin):
    """Display to inspect the coefficients of linear models.

    Parameters
    ----------
    coefficients : DataFrame | list[DataFrame]
        The coefficients data to display. The columns are:

        - `estimator`
        - `split`
        - `feature`
        - `label` or `output` (classification vs. regression)
        - `coefficients`

    report_type : {"estimator", "cross-validation", "comparison-estimator", \
            "comparison-cross-validation"}
        Report type from which the display is created.

    Attributes
    ----------
    ax_ : matplotlib Axes
        Axes with the different matplotlib axis.

    figure_ : matplotlib Figure
        Figure containing the plot.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skore import EstimatorReport, train_test_split
    >>> iris = load_iris(as_frame=True)
    >>> X, y = iris.data, iris.target
    >>> y = iris.target_names[y]
    >>> split_data = train_test_split(
    ...     X=X, y=y, random_state=0, as_dict=True, shuffle=True
    ... )
    >>> report = EstimatorReport(LogisticRegression(), **split_data)
    >>> display = report.feature_importance.coefficients()
    >>> display.frame()
                  feature       label  coefficients
    0           Intercept      setosa      9.2...
    1   sepal length (cm)      setosa     -0.4...
    2    sepal width (cm)      setosa      0.8...
    3   petal length (cm)      setosa     -2.3...
    4    petal width (cm)      setosa     -0.9...
    5           Intercept  versicolor      1.7...
    6   sepal length (cm)  versicolor      0.5...
    7    sepal width (cm)  versicolor     -0.2...
    8   petal length (cm)  versicolor     -0.2...
    9    petal width (cm)  versicolor     -0.7...
    10          Intercept   virginica    -11.0...
    11  sepal length (cm)   virginica     -0.1...
    12   sepal width (cm)   virginica     -0.5...
    13  petal length (cm)   virginica      2.5...
    14   petal width (cm)   virginica      1.7...
    """

    _default_barplot_kwargs: dict[str, Any] = {"palette": "tab10"}
    _default_boxplot_kwargs: dict[str, Any] = {
        "palette": "tab10",
        "vert": False,
        "whis": 100_000,
        **BOXPLOT_STYLE,
    }

    def __init__(self, *, coefficients: pd.DataFrame, report_type: ReportType):
        self.coefficients = coefficients
        self.report_type = report_type

    def frame(self):
        """Get the coefficients in a dataframe format.

        The returned dataframe is not going to contain constant columns or columns
        containing only NaN values.

        Returns
        -------
        DataFrame
            Dataframe containing the coefficients of the linear model.

        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import EstimatorReport, train_test_split
        >>> iris = load_iris(as_frame=True)
        >>> X, y = iris.data, iris.target
        >>> y = iris.target_names[y]
        >>> split_data = train_test_split(
        ...     X=X, y=y, random_state=0, as_dict=True, shuffle=True
        ... )
        >>> report = EstimatorReport(LogisticRegression(), **split_data)
        >>> display = report.feature_importance.coefficients()
        >>> display.frame()
                    feature       label  coefficients
        0           Intercept      setosa      9.2...
        1   sepal length (cm)      setosa     -0.4...
        2    sepal width (cm)      setosa      0.8...
        3   petal length (cm)      setosa     -2.3...
        4    petal width (cm)      setosa     -0.9...
        5           Intercept  versicolor      1.7...
        6   sepal length (cm)  versicolor      0.5...
        7    sepal width (cm)  versicolor     -0.2...
        8   petal length (cm)  versicolor     -0.2...
        9    petal width (cm)  versicolor     -0.7...
        10          Intercept   virginica    -11.0...
        11  sepal length (cm)   virginica     -0.1...
        12   sepal width (cm)   virginica     -0.5...
        13  petal length (cm)   virginica      2.5...
        14   petal width (cm)   virginica      1.7...
        """
        if self.report_type == "estimator":
            columns_to_drop = ["estimator", "split"]
        elif self.report_type == "cross-validation":
            columns_to_drop = ["estimator"]
        elif self.report_type == "comparison-estimator":
            columns_to_drop = ["split"]
        elif self.report_type == "comparison-cross-validation":
            columns_to_drop = []
        else:
            raise TypeError(f"Unexpected report type: {self.report_type!r}")

        if self.coefficients["label"].isna().all():
            # regression problem
            columns_to_drop.append("label")
        if self.coefficients["output"].isna().all():
            # classification problem
            columns_to_drop.append("output")

        return self.coefficients.drop(columns=columns_to_drop)

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        subplots_by: Literal["estimator", "label", "output"] | None = None,
        barplot_kwargs: dict[str, Any] | None = None,
        boxplot_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Plot the coefficients for the different features.

        An instance of this class should be created by
        `report.feature_importance.coefficients()`.

        Parameters
        ----------
        subplots_by : {"estimator", "label", "output"}, default=None
            The column to use for subplotting and dividing the coefficients into
            subplots. If `None`, an automatic choice is made depending on the type of
            reports at hand.

        barplot_kwargs : dict, default=None
            Keyword arguments to be passed to seaborn's :func:`seaborn.barplot` for
            rendering the coefficients with an :class:`~skore.EstimatorReport` or
            :class:`~skore.ComparisonReport` of :class:`~skore.EstimatorReport`.

        boxplot_kwargs : dict, default=None
            Keyword arguments to be passed to seaborn's :func:`seaborn.boxplot` for
            rendering the coefficients with a :class:`~skore.CrossValidationReport` or
            :class:`~skore.ComparisonReport` of :class:`~skore.CrossValidationReport`.

        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import EstimatorReport, train_test_split
        >>> iris = load_iris(as_frame=True)
        >>> X, y = iris.data, iris.target
        >>> y = iris.target_names[y]
        >>> split_data = train_test_split(
        ...     X=X, y=y, random_state=0, as_dict=True, shuffle=True
        ... )
        >>> report = EstimatorReport(LogisticRegression(), **split_data)
        >>> display = report.feature_importance.coefficients()
        >>> display.plot()
        """
        return self._plot(
            subplots_by=subplots_by,
            barplot_kwargs=barplot_kwargs,
            boxplot_kwargs=boxplot_kwargs,
        )

    def _plot_matplotlib(
        self,
        *,
        subplots_by: Literal["estimator", "label", "output"] | None = None,
        barplot_kwargs: dict[str, Any] | None = None,
        boxplot_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Dispatch the plotting function for matplotlib backend."""
        # make copy of the dictionary since we are going to pop some keys later
        if barplot_kwargs is None:
            barplot_kwargs = self._default_barplot_kwargs.copy()
        else:
            barplot_kwargs = {**self._default_barplot_kwargs, **barplot_kwargs}
        if boxplot_kwargs is None:
            boxplot_kwargs = self._default_boxplot_kwargs.copy()
        else:
            boxplot_kwargs = {**self._default_boxplot_kwargs, **boxplot_kwargs}

        if self.report_type == "estimator":
            return self._plot_single_estimator(
                subplots_by=subplots_by,
                plot_function=sns.barplot,
                plot_function_kwargs=barplot_kwargs,
            )
        elif self.report_type == "cross-validation":
            return self._plot_single_estimator(
                subplots_by=subplots_by,
                plot_function=sns.boxplot,
                plot_function_kwargs=boxplot_kwargs,
            )
        elif self.report_type == "comparison-estimator":
            return self._plot_comparison(
                subplots_by=subplots_by,
                plot_function=sns.barplot,
                plot_function_kwargs=barplot_kwargs,
            )
        elif self.report_type == "comparison-cross-validation":
            return self._plot_comparison(
                subplots_by=subplots_by,
                plot_function=sns.boxplot,
                plot_function_kwargs=boxplot_kwargs,
            )
        else:
            raise TypeError(f"Unexpected report type: {self.report_type!r}")

    @staticmethod
    def _get_figsize(
        *,
        frame: pd.DataFrame,
        subplots_by: Literal["estimator", "label", "output"] | None = None,
        hue: str | None = None,
        has_same_features: bool = True,
    ) -> tuple[float, float]:
        """Get the figure size for the plot based on what we try to plot.

        In short, `subplots_by` will indicate that we need several columns. The height
        will depend on the max number of features across all estimators and the number
        of groups of hue.

        Parameters
        ----------
        frame : DataFrame
            The frame containing the data to plot.
        subplots_by : {"estimator", "label", "output"} or None, default=None
            The column to use for subplotting and dividing the coefficients into
            subplots. If `None`, there is a single axis.
        hue : str or None, default=None
            The column used to group by for the color.
        has_same_features : bool, default=True
            Whether the features are the same across all estimators. If `False`, we need
            to find the largest number of features across all estimators.

        Returns
        -------
        tuple[float, float]
            The figure size.
        """
        min_width, min_height, min_height_per_item = 6.4, 4.8, 0.3

        # for comparison reports where we don't have the same features, we need to
        # find the largest number of features across all estimators
        if has_same_features:
            n_features = frame["feature"].nunique()
        else:
            n_features = max(
                [
                    group_frame["feature"].nunique()
                    for _, group_frame in frame.groupby("estimator")
                ]
            )
        # with hue, it means that we will have several bars of boxes for a single
        # feature: we need to find this number of groups
        n_hue_groups = 1 if hue is None else frame[hue].nunique()
        height = max(min_height, min_height_per_item * n_features * n_hue_groups)

        if subplots_by is None:
            width = min_width
        else:
            ncols = frame[subplots_by].nunique()
            width = min_width * ncols

        return (width, height)

    @staticmethod
    def _decorate_matplotlib_axis(
        *, ax: plt.Axes, add_background_features: bool = False, n_features: int
    ) -> None:
        """Decorate the matplotlib axis.

        Parameters
        ----------
        ax : plt.Axes
            The matplotlib axis to decorate.
        add_background_features : bool, default=False
            Whether to add a background color for each group of features.
        n_features : int
            The number of features to displayed.
        """
        ax.axvline(x=0, color=".5", linestyle="--")
        ax.set(xlabel="Magnitude of coefficient", ylabel="")
        _despine_matplotlib_axis(
            ax,
            axis_to_despine=("top", "right", "left"),
            remove_ticks=True,
            x_range=None,
            y_range=None,
        )
        if add_background_features:
            for feature_idx in range(n_features):
                if feature_idx % 2 == 0:
                    ax.axhspan(
                        feature_idx - 0.5,
                        feature_idx + 0.5,
                        color="lightgray",
                        alpha=0.1,
                        zorder=0,
                    )

    @staticmethod
    def _set_legend(*, ax: plt.Axes, title: str) -> None:
        """Set the legend title and location."""
        legend = ax.get_legend()
        if legend:
            legend.set_title(title)
            legend.set_loc("best")

    @staticmethod
    def _get_columns_to_groupby(*, frame: pd.DataFrame) -> list[str]:
        """Get the available columns from which to group by."""
        columns_to_groupby = list[str]()
        if "estimator" in frame.columns:
            columns_to_groupby.append("estimator")
        if "label" in frame.columns:
            columns_to_groupby.append("label")
        if "output" in frame.columns:
            columns_to_groupby.append("output")
        return columns_to_groupby

    def _plot_single_estimator(
        self,
        *,
        plot_function: Callable,
        plot_function_kwargs: dict,
        subplots_by: Literal["estimator", "label", "output"] | None = None,
    ) -> None:
        """Plot the coefficients for an `EstimatorReport` or a `CrossValidationReport`.

        An `EstimatorReport` will use a bar plot while a `CrossValidationReport` will
        use a box plot.

        Parameters
        ----------
        plot_function : Callable
            The function to use to plot the coefficients.
        plot_function_kwargs : dict
            The keyword arguments to pass to the plot function.
        subplots_by : {"estimator", "label", "output"}, default=None
            The column to use for subplotting and dividing the coefficients into
            subplots. If `None`, an automatic choice is made depending on the type of
            reports at hand.
        """
        frame, name = self.frame(), self.coefficients["estimator"].unique()[0]

        # {"label"} or {"output"} or {}
        columns_to_groupby = self._get_columns_to_groupby(frame=frame)
        if subplots_by is not None and not len(columns_to_groupby):
            raise ValueError("No columns to group by.")
        elif subplots_by is not None and subplots_by not in columns_to_groupby:
            raise ValueError(
                f"Column {subplots_by} not found in the frame. It should be one "
                f"of {', '.join(columns_to_groupby)}."
            )

        if subplots_by is None:
            hue = None if not len(columns_to_groupby) else columns_to_groupby[0]
            palette = plot_function_kwargs.pop("palette")
            palette = palette if hue is not None else None
            self.figure_, self.ax_ = plt.subplots(
                figsize=self._get_figsize(
                    frame=frame,
                    subplots_by=subplots_by,
                    hue=hue,
                    has_same_features=True,
                )
            )

            plot_function(
                data=frame,
                x="coefficients",
                y="feature",
                hue=hue,
                palette=palette,
                ax=self.ax_,
                **plot_function_kwargs,
            )
            add_background_features = hue is not None and plot_function == sns.boxplot
            self._decorate_matplotlib_axis(
                ax=self.ax_,
                add_background_features=add_background_features,
                n_features=frame["feature"].nunique(),
            )
            self.ax_.set_title(f"{name}")
            if hue is not None:
                self._set_legend(ax=self.ax_, title=hue.capitalize())
        else:
            hue = None
            # we don't need the palette and we are at risk of raising an error or
            # deprecation warning if passing palette without a hue
            plot_function_kwargs.pop("palette", None)

            self.figure_, self.ax_ = plt.subplots(
                ncols=frame[subplots_by].nunique(),
                sharex=True,
                sharey=True,
                figsize=self._get_figsize(
                    frame=frame,
                    subplots_by=subplots_by,
                    hue=hue,
                    has_same_features=True,
                ),
            )

            axes = cast(np.ndarray, self.ax_)
            for ax, (group, group_frame) in zip(
                axes.flatten(), frame.groupby(by=subplots_by), strict=True
            ):
                plot_function(
                    data=group_frame,
                    x="coefficients",
                    y="feature",
                    ax=ax,
                    **plot_function_kwargs,
                )
                self._decorate_matplotlib_axis(
                    ax=ax,
                    add_background_features=False,
                    n_features=group_frame["feature"].nunique(),
                )
                ax.set_title(f"{name} - {subplots_by.capitalize()}: {group}")

    @staticmethod
    def _has_same_features(*, frame: pd.DataFrame) -> bool:
        """Check if the features are the same across all estimators."""
        grouped = {
            name: group["feature"].sort_values().tolist()
            for name, group in frame.groupby("estimator")
        }
        _, reference_features = grouped.popitem()
        for group_features in grouped.values():
            if group_features != reference_features:
                return False
        return True

    def _plot_comparison(
        self,
        *,
        plot_function: Callable,
        plot_function_kwargs: dict,
        subplots_by: Literal["estimator", "label", "output"] | None = None,
    ) -> None:
        """Plot the coefficients for a `ComparisonReport`.

        Parameters
        ----------
        plot_function : Callable
            The function to use to plot the coefficients.
        plot_function_kwargs : dict
            The keyword arguments to pass to the plot function.
        subplots_by : {"estimator", "label", "output"}, default=None
            The column to use for subplotting and dividing the coefficients into
            subplots. If `None`, an automatic choice is made depending on the type of
            reports at hand.
        """
        frame = self.frame()
        # help mypy to understand the following variable types
        hue: str | None = None
        palette: str | None = None

        # {"estimator", "label"} or {"estimator", "output"} or {"estimator"}
        columns_to_groupby = self._get_columns_to_groupby(frame=frame)
        if subplots_by is not None and subplots_by not in columns_to_groupby:
            raise ValueError(
                f"Column {subplots_by} not found in the frame. It should be one "
                f"of {', '.join(columns_to_groupby)}."
            )

        has_same_features = self._has_same_features(frame=frame)
        if (frame.columns.isin(["label", "output"]).any() and subplots_by is None) or (
            not has_same_features and subplots_by is None
        ):
            # default fallback on subplots by estimator
            # case 1: multiclass classification or multi-output regression
            # therefore, too many information to display on a single plot, by default
            # group by estimator
            # case 2: features cannot be compared across estimators and we therefore
            # need to subplots by estimator
            subplots_by = "estimator"

        if subplots_by is None:
            hue, palette = columns_to_groupby[0], plot_function_kwargs.pop("palette")
            self.figure_, self.ax_ = plt.subplots(
                figsize=self._get_figsize(
                    frame=frame,
                    subplots_by=subplots_by,
                    hue=hue,
                    has_same_features=True,
                )
            )

            plot_function(
                data=frame,
                x="coefficients",
                y="feature",
                hue=hue,
                palette=palette,
                ax=self.ax_,
                **plot_function_kwargs,
            )
            self._decorate_matplotlib_axis(
                ax=self.ax_,
                add_background_features=plot_function == sns.boxplot,
                n_features=frame["feature"].nunique(),
            )
            if hue is not None:
                self._set_legend(ax=self.ax_, title=hue.capitalize())
        else:
            # infer if we should group by another column using hue
            hue_groupby = [col for col in columns_to_groupby if col != subplots_by]
            hue = hue_groupby[0] if len(hue_groupby) else None
            palette = plot_function_kwargs.pop("palette")
            palette = palette if hue is not None else None

            if not has_same_features and hue == "estimator":
                raise ValueError(
                    "The estimators have different features and should be plotted on "
                    "different axis using `subplots_by='estimator'`."
                )

            self.figure_, self.ax_ = plt.subplots(
                ncols=frame[subplots_by].nunique(),
                sharex=True,
                sharey=has_same_features,
                figsize=self._get_figsize(
                    frame=frame,
                    subplots_by=subplots_by,
                    hue=hue,
                    has_same_features=has_same_features,
                ),
            )

            axes = cast(np.ndarray, self.ax_)
            for ax, (group, group_frame) in zip(
                axes.flatten(), frame.groupby(by=subplots_by), strict=True
            ):
                plot_function(
                    data=group_frame,
                    x="coefficients",
                    y="feature",
                    hue=hue,
                    palette=palette,
                    ax=ax,
                    **plot_function_kwargs,
                )
                self._decorate_matplotlib_axis(
                    ax=ax,
                    add_background_features=plot_function == sns.boxplot,
                    n_features=group_frame["feature"].nunique(),
                )
                if hue is not None:
                    self._set_legend(ax=ax, title=hue.capitalize())
                ax.set_title(f"{subplots_by.capitalize()}: {group}")

    @classmethod
    def _compute_data_for_display(
        cls,
        *,
        estimators: Sequence[BaseEstimator],
        names: list[str],
        splits: list[int | float],
        report_type: ReportType,
    ) -> "CoefficientsDisplay":
        """Compute the data for the display.

        Parameters
        ----------
        estimators : list of estimator
            The estimators to compute the data for.
        names : list of str
            The names of the estimators.
        splits : list of int or np.nan
            The splits to compute the data for.
        report_type : {"estimator", "cross-validation", "comparison-estimator", \
                "comparison-cross-validation"}
            The type of report to compute the data for.

        Returns
        -------
        CoefficientsDisplay
            The data for the display.
        """
        feature_names, est_names, coefficients, split_indices = [], [], [], []
        for estimator, name, split in zip(estimators, names, splits, strict=True):
            if isinstance(estimator, Pipeline):
                preprocessor, predictor = estimator[:-1], estimator[-1]
            else:
                preprocessor, predictor = None, estimator

            if isinstance(predictor, TransformedTargetRegressor):
                predictor = predictor.regressor_

            coef = np.atleast_2d(predictor.coef_).T
            intercept = np.atleast_2d(predictor.intercept_)
            if coef.shape[1] != intercept.shape[1]:
                # it happens that with `fit_intercept=False` and a multi-output
                # regression problem, intercept is a single float. Thus, we need to
                # repeat it for each output
                intercept = np.repeat(intercept, coef.shape[1], axis=1)
            coefficients.append(np.concatenate([intercept, coef]))

            feat_names = ["Intercept"] + _get_feature_names(
                predictor, transformer=preprocessor, n_features=coef.shape[0]
            )
            feature_names.extend(feat_names)
            est_names.extend([name] * len(feat_names))
            split_indices.extend([split] * len(feat_names))

        index = pd.DataFrame(
            {
                "estimator": est_names,
                "split": split_indices,
                "feature": feature_names,
            }
        )

        if coef.shape[1] == 1:
            # binary or single output regression
            columns, require_melting = ["coefficients"], False
            index["label"], index["output"] = np.nan, np.nan
        else:
            require_melting = True
            if is_classifier(predictor):
                # multi-class classification
                columns, var_name = predictor.classes_.tolist(), "label"
                index["output"] = np.nan
            else:
                # multi-output regression
                columns, var_name = [f"{i}" for i in range(coef.shape[1])], "output"
                index["label"] = np.nan
            id_vars, value_name = index.columns.tolist(), "coefficients"

        coefficients = pd.DataFrame(
            np.concatenate(coefficients, axis=0), columns=columns
        )
        coefficients = pd.concat([index, coefficients], axis=1)
        if require_melting:
            # melt the coefficients and ensure alignment with the label/output, split
            # feature names, and estimator names
            coefficients = coefficients.melt(
                id_vars=id_vars,
                value_vars=columns,
                var_name=var_name,
                value_name=value_name,
            )

        return cls(coefficients=coefficients, report_type=report_type)
