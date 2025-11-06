import matplotlib.pyplot as plt

from skore._sklearn._plot.base import DisplayMixin


class FeatureImportanceCoefficientsDisplay(DisplayMixin):
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

    def __init__(self, report_type, coefficients):
        self.report_type = report_type
        self.coefficients = coefficients

    def frame(self):
        """Return coefficients as a DataFrame.

        Returns
        -------
        pd.DataFrame
            The structure of the returned frame depends on the underlying report type:

            - If an :class:`EstimatorReport`, a single column "Coefficient", with the
              index being the feature names.

            - If a :class:`CrossValidationReport`, the columns are the feature names,
              and the index is the respective split number.

            - If a :class:`ComparisonReport`, the columns are the models passed in the
              report, with the index being the feature names.
        """
        if self.report_type == "estimator":
            return self._frame_estimator_report()
        elif self.report_type == "cross-validation":
            return self._frame_cross_validation_report()
        else:
            return self._frame_comparison_report()

    def _frame_estimator_report(self):
        return self.coefficients

    def _frame_cross_validation_report(self):
        return self.coefficients

    def _frame_comparison_report(self):
        import pandas as pd

        return pd.concat(self.coefficients, axis=1)

    @DisplayMixin.style_plot
    def plot(self, **kwargs) -> None:
        """Plot the coefficients of linear models.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to be passed to the plot method.
        """
        return self._plot(**kwargs)

    def _style_plot_matplotlib(self, ax, title=None, legend=True):
        if title:
            ax.set_title(title)
        if legend:
            ax.legend(loc="best", bbox_to_anchor=(1, 1), borderaxespad=1)
        ax.grid(False)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)

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
        self.coefficients.plot.barh(ax=self.ax_)
        self._style_plot_matplotlib(self.ax_, title="Coefficients")
        self.figure_.tight_layout()
        plt.show()

    def _plot_cross_validation_report(self):
        self.figure_, self.ax_ = plt.subplots()
        self.coefficients.boxplot(ax=self.ax_, vert=False)
        self._style_plot_matplotlib(
            self.ax_, title="Coefficient variance across CV splits", legend=None
        )
        self.figure_.tight_layout()
        plt.show()

    def _plot_comparison_report_estimator(self):
        self.figure_, self.ax_ = plt.subplots(
            nrows=1,
            ncols=len(self.coefficients),
            figsize=(5 * len(self.coefficients), 6),
            squeeze=False,
        )
        self.ax_ = self.ax_.flatten()
        self.figure_.suptitle("Coefficients")
        for ax, coef_frame in zip(self.ax_, self.coefficients, strict=False):
            coef_frame.plot.barh(ax=ax)
            self._style_plot_matplotlib(ax, title=None)
        self.figure_.tight_layout()
        plt.show()

    def _plot_comparison_report_cross_validation(self):
        self.figure_, self.ax_ = plt.subplots(
            nrows=1,
            ncols=len(self.coefficients),
            figsize=(5 * len(self.coefficients), 6),
            squeeze=False,
        )
        self.ax_ = self.ax_.flatten()
        for ax, coef_frame in zip(self.ax_, self.coefficients, strict=False):
            coef_frame.boxplot(ax=ax, vert=False)
            model_name = coef_frame.columns[0].split("__")[0]
            self._style_plot_matplotlib(
                ax,
                title=f"{model_name} Coefficients across splits",
                legend=None,
            )
        self.figure_.tight_layout()
        plt.show()
