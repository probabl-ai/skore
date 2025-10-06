import matplotlib.pyplot as plt

from skore._sklearn._plot.base import DisplayMixin


class FeatureImportanceCoefficientsDisplay(DisplayMixin):
    """Feature importance display.

    Each report type produces its own output frame and plot.

    Parameters
    ----------
    parent : {"estimator", "cross-validation", "comparison-estimator",
            "comparison-cross-validation"}
        Report type from which the display is created.

    coefficient_data : DataFrame | list[DataFrame]
        The ROC AUC data to display. The columns are

    Attributes
    ----------
    ax_ : matplotlib Axes
        Axes with the different matplotlib axis.

    figure_ : matplotlib Figure
        Figure containing the plot.

    Methods
    -------
    frame() -> DataFrame
        The coefficients as a dataframe.

    plot() -> NoneType
        A plot of the coefficients.

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

    def __init__(self, parent, coefficient_data):
        self._parent = parent
        self._coefficient_data = coefficient_data

    def frame(self):
        """Return coefficients as a DataFrame.

        Returns
        -------
        pd.DataFrame
            The structure of the returned frame depends on the underlying report type:

            - If an ``EstimatorReport``, a single column
            "Coefficient", with the index being the feature names.

            - If a ``CrossValidationReport``, the columns are
            the feature names, and the index is the respective split number.

            - If a ``ComparisonReport``, the columns are the
            models passed in the report, with the index being the feature names.
        """
        if self._parent == "estimator":
            return self._frame_estimator_report()
        elif self._parent == "cross-validation":
            return self._frame_cross_validation_report()
        else:
            return self._frame_comparison_report()

    def _frame_estimator_report(self):
        return self._coefficient_data

    def _frame_cross_validation_report(self):
        return self._coefficient_data

    def _frame_comparison_report(self):
        import pandas as pd

        return pd.concat(self._coefficient_data, axis=1)

    @DisplayMixin.style_plot
    def plot(self, **kwargs) -> None:
        """Plot the coefficients of linear models.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to be passed to the plot method.
        """
        return self._plot(**kwargs)

    def _plot_matplotlib(self, **kwargs):
        if self._parent == "estimator":
            return self._plot_estimator_report()
        elif self._parent == "cross-validation":
            return self._plot_cross_validation_report()
        else:
            return self._plot_comparison_report()

    def _plot_estimator_report(self):
        self.figure_, self.ax_ = plt.subplots()
        self._coefficient_data.plot.barh(ax=self.ax_)
        self.ax_.set_title("Coefficients")
        self.ax_.legend(loc="best", bbox_to_anchor=(1, 1), borderaxespad=1)
        self.ax_.grid(False)
        for spine in ["top", "right", "left"]:
            self.ax_.spines[spine].set_visible(False)
        self.ax_.tick_params(axis="y", length=0)
        self.figure_.tight_layout()
        plt.show()

    def _plot_cross_validation_report(self):
        self.figure_, self.ax_ = plt.subplots()
        self._coefficient_data.boxplot(ax=self.ax_, vert=False)
        self.ax_.set_title("Coefficient variance across CV splits")
        self.ax_.grid(False)
        for spine in ["top", "right", "left"]:
            self.ax_.spines[spine].set_visible(False)
        self.ax_.tick_params(axis="y", length=0)
        self.figure_.tight_layout()
        plt.show()

    def _plot_comparison_report(self):
        self.figure_, self.ax_ = plt.subplots(
            nrows=1,
            ncols=len(self._coefficient_data),
            figsize=(5 * len(self._coefficient_data), 6),
            squeeze=False,
        )
        self.ax_ = self.ax_.flatten()

        if self._parent == "comparison-estimator":
            self.figure_.suptitle("Coefficients")
            for ax, coef_frame in zip(self.ax_, self._coefficient_data, strict=False):
                coef_frame.plot.barh(ax=ax)
                ax.legend(loc="best", bbox_to_anchor=(1, 1), borderaxespad=1)
                ax.grid(False)
                for spine in ["top", "right", "left"]:
                    ax.spines[spine].set_visible(False)
                ax.tick_params(axis="y", length=0)

        elif self._parent == "comparison-cross-validation":
            for ax, coef_frame in zip(self.ax_, self._coefficient_data, strict=False):
                coef_frame.boxplot(ax=ax, vert=False)
                ax.set_title(
                    f"{coef_frame.columns[0].split('__')[0]} Coefficients across splits"
                )
                ax.grid(False)
                for spine in ["top", "right", "left"]:
                    ax.spines[spine].set_visible(False)
                ax.tick_params(axis="y", length=0)
        else:
            raise TypeError(f"Unexpected report type: {self._parent}")

        self.figure_.tight_layout()
        plt.show()
