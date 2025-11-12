from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend import Legend
from sklearn.base import BaseEstimator, is_classifier

from skore._sklearn._plot.base import BOXPLOT_STYLE, DisplayMixin
from skore._sklearn._plot.utils import _despine_matplotlib_axis
from skore._sklearn.feature_names import _get_feature_names
from skore._sklearn.types import ReportType


class CoefficientsDisplay(DisplayMixin):
    """Feature importance display.

    Each report type produces its own output frame and plot.

    Parameters
    ----------
    report_type : {"estimator", "cross-validation", "comparison-estimator", \
            "comparison-cross-validation"}
        Report type from which the display is created.

    coefficients : DataFrame | list[DataFrame]
        The coefficients data to display.

    Attributes
    ----------
    ax_ : matplotlib Axes
        Axes with the different matplotlib axis.

    figure_ : matplotlib Figure
        Figure containing the plot.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.linear_model import LinearRegression
    >>> from skore import train_test_split
    >>> from skore import EstimatorReport
    >>> X, y = load_diabetes(return_X_y=True)
    >>> split_data = train_test_split(
    >>>     X=X, y=y, random_state=0, as_dict=True, shuffle=False
    >>> )
    >>> report = EstimatorReport(LinearRegression(), **split_data)
    >>> display = report.feature_importance.coefficients()
    >>> display.plot()
    >>> display.frame()
                Coefficient
    Intercept	151.487952
    Feature #0	-11.861904
    Feature #1	-238.445509
    Feature #2	505.395493
    Feature #3	298.977119
    ...         ...
    """

    def __init__(self, *, coefficients: pd.DataFrame, report_type: ReportType):
        self.coefficients = coefficients
        self.report_type = report_type

    def frame(self):
        if self.report_type == "estimator":
            columns_to_drop = ["estimator_name", "split"]
        elif self.report_type == "cross-validation":
            columns_to_drop = ["estimator_name"]
        elif self.report_type == "comparison-estimator":
            columns_to_drop = ["split"]
        elif self.report_type == "comparison-cross-validation":
            columns_to_drop = []
        else:
            raise TypeError(f"Unexpected report type: {self.report_type!r}")

        return self.coefficients.drop(columns=columns_to_drop)

    @DisplayMixin.style_plot
    def plot(self, **kwargs) -> None:
        return self._plot(**kwargs)

    def _decorate_matplotlib_axis(self):
        self.ax_.axvline(x=0, color=".5", linestyle="--")
        self.ax_.set(xlabel="Magnitude of coefficient", ylabel="")
        _despine_matplotlib_axis(
            self.ax_,
            axis_to_despine=("top", "right", "left"),
            remove_ticks=True,
            x_range=None,
            y_range=None,
        )

    @staticmethod
    def _set_legend(legend: Legend):
        legend.set_title("Estimator")
        legend.set_loc("upper right")

    def _plot_matplotlib(self, **kwargs):
        if self.report_type == "estimator":
            return self._plot_estimator_report()
        elif self.report_type == "cross-validation":
            return self._plot_cross_validation_report()
        elif self.report_type == "comparison-estimator":
            return self._plot_comparison_report_estimator()
        elif self.report_type == "comparison-cross-validation":
            return self._plot_comparison_report_cross_validation()
        else:
            raise TypeError(f"Unexpected report type: {self.report_type!r}")

    def _plot_estimator_report(self):
        self.figure_, self.ax_ = plt.subplots()
        sns.barplot(data=self.frame(), x="coefficients", y="feature_name", ax=self.ax_)
        self._decorate_matplotlib_axis()

    def _plot_cross_validation_report(self):
        self.figure_, self.ax_ = plt.subplots()
        sns.boxplot(
            data=self.frame(),
            x="coefficients",
            y="feature_name",
            ax=self.ax_,
            vert=False,
            whis=100_000,
            **BOXPLOT_STYLE,
        )
        self._decorate_matplotlib_axis()

    def _plot_comparison_report_estimator(self):
        self.figure_, self.ax_ = plt.subplots()
        sns.barplot(
            data=self.frame(),
            x="coefficients",
            y="feature_name",
            hue="estimator_name",
            ax=self.ax_,
        )
        self._decorate_matplotlib_axis()
        self._set_legend(self.ax_.get_legend())

    def _plot_comparison_report_cross_validation(self):
        self.figure_, self.ax_ = plt.subplots()
        sns.boxplot(
            data=self.frame(),
            x="coefficients",
            y="feature_name",
            hue="estimator_name",
            ax=self.ax_,
            vert=False,
            whis=100_000,
            **BOXPLOT_STYLE,
        )
        self._decorate_matplotlib_axis()
        self._set_legend(self.ax_.get_legend())

    @classmethod
    def _compute_data_for_display(
        cls,
        *,
        estimators: Sequence[BaseEstimator],
        names: list[str],
        splits: list[int | None],
        report_type: ReportType,
    ):
        feature_names, est_names, coefficients, split_indices = [], [], [], []
        for estimator, name, split in zip(estimators, names, splits, strict=True):
            try:
                coef = np.atleast_2d(estimator.coef_).T
                intercept = np.atleast_2d(estimator.intercept_)
            except AttributeError:
                # TransformedTargetRegressor() is a meta-estimator exposing `regressor_`
                # instead of exposing directly the coefficients
                coef = np.atleast_2d(estimator.regressor_.coef_).T
                intercept = np.atleast_2d(estimator.regressor_.intercept_)

            feat_names = _get_feature_names(estimator, n_features=coef.shape[0])
            if intercept is None:
                coefficients.append(coef)
                feature_names.extend(feat_names)
            else:
                coefficients.append(np.concatenate([intercept, coef]))
                feat_names.insert(0, "Intercept")
                feature_names.extend(feat_names)
            est_names.extend([name] * len(feat_names))
            split_indices.extend([split] * len(feat_names))

        if coef.shape[1] == 1:
            columns = ["coefficients"]
        elif is_classifier(estimator):
            columns = [f"class_{i}" for i in range(coef.shape[1])]
        else:
            columns = [f"target_{i}" for i in range(coef.shape[1])]

        info = pd.DataFrame(
            {
                "estimator_name": est_names,
                "split": split_indices,
                "feature_name": feature_names,
            }
        )
        coefficients = pd.DataFrame(
            np.concatenate(coefficients, axis=0), columns=columns
        )
        return cls(
            coefficients=pd.concat([info, coefficients], axis=1),
            report_type=report_type,
        )
