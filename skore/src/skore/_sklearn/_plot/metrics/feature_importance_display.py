import matplotlib.pyplot as plt

from skore._sklearn._plot.base import Display
from skore._sklearn._plot.style import StyleDisplayMixin
from skore._sklearn._plot.utils import HelpDisplayMixin, PlotBackendMixin


class FeatureImportanceDisplay(
    HelpDisplayMixin, StyleDisplayMixin, PlotBackendMixin, Display
):
    """Feature importance display.

    Each report type produces its own output frame and plot.

    Parameters
    ----------
    _parent : EstimatorReport | CrossValidationReport | ComparisonReport
        Report type from which the display is created.

    _coefficient_data : DataFrame | list[DataFrame]
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

    def __init__(self, _parent, _coefficient_data):
        self._parent = _parent
        self._coefficient_data = _coefficient_data

    def frame(self):
        """Return coefficients as a DataFrame.

        Returns
        -------
        pd.DataFrame
            The structure of the returned DataFrame depends on the type of the underlying report:

            - If an ``EstimatorReport``, a single column
            "Coefficient", with the index being the feature names.

            - If a ``CrossValidationReport``, the columns are
            the feature names, and the index is the respective split number.

            - If a ``ComparisonReport``, the columns are the
            models passed in the report, with the index being the feature names.
        """
        from skore import ComparisonReport, CrossValidationReport, EstimatorReport

        if isinstance(self._parent, EstimatorReport):
            return self._frame_estimator_report()
        elif isinstance(self._parent, CrossValidationReport):
            return self._frame_cross_validation_report()
        elif isinstance(self._parent, ComparisonReport):
            return self._frame_comparison_report()
        else:
            raise TypeError(f"Unrecognised report type: {self._parent}")

    def _frame_estimator_report(self):
        return self._coefficient_data

    def _frame_cross_validation_report(self):
        return self._coefficient_data

    def _frame_comparison_report(self):
        import pandas as pd

        return pd.concat(self._coefficient_data, axis=1)

    @StyleDisplayMixin.style_plot
    def _plot_matplotlib(self, **kwargs):
        from skore._sklearn._comparison import ComparisonReport
        from skore._sklearn._cross_validation import CrossValidationReport
        from skore._sklearn._estimator import EstimatorReport

        if isinstance(self._parent, EstimatorReport):
            return self._plot_estimator_report()
        elif isinstance(self._parent, CrossValidationReport):
            return self._plot_cross_validation_report()
        elif isinstance(self._parent, ComparisonReport):
            return self._plot_comparison_report()
        else:
            raise TypeError(f"Unrecognised report type: {self._parent}")

    def _plot_estimator_report(self):
        self.figure_, self.ax_ = plt.subplots()
        self._coefficient_data.plot.bar(ax=self.ax_)
        self.ax_.set_title(f"{self._parent.estimator_name_} Coefficients")
        self.ax_.legend(loc="best", bbox_to_anchor=(1, 1), borderaxespad=1)
        self.figure_.tight_layout()
        plt.show()

    def _plot_cross_validation_report(self):
        self.figure_, self.ax_ = plt.subplots()
        self._coefficient_data.boxplot(ax=self.ax_)
        self.ax_.set_title("Coefficient variance across CV splits")
        self.figure_.tight_layout()
        plt.show()

    def _plot_comparison_report(self):
        from skore import CrossValidationReport, EstimatorReport

        if isinstance(self._parent.reports_[0], EstimatorReport):
            for coef_frame in self._coefficient_data:
                self.figure_, self.ax_ = plt.subplots()
                coef_frame.plot.bar(ax=self.ax_)
                self.ax_.legend(loc="best", bbox_to_anchor=(1, 1), borderaxespad=1)

                if len(self._coefficient_data) == 1 or len(coef_frame.columns) > 1:
                    # If there's only one DataFrame, or if the plot includes
                    # multiple models with the same features, use a combined title
                    # like "Model 1 vs Model 2 Coefficients".
                    #
                    # For example, if 3 reports are passed and 2 share the same
                    # features, those two are plotted together with a combined
                    # title, while the third goes to the else clause and gets its
                    # title.
                    self.ax_.set_title(
                        f"{' vs '.join(coef_frame.columns)} Coefficients"
                    )
                else:
                    self.ax_.set_title(f"{coef_frame.columns[0]} Coefficients")
                self.figure_.tight_layout()
                plt.show()
        elif isinstance(self._parent.reports_[0], CrossValidationReport):
            for coef_frame in self._coefficient_data:
                self.figure_, self.ax_ = plt.subplots()
                coef_frame.boxplot(ax=self.ax_)
                self.ax_.set_title(
                    f"{coef_frame.columns[0].split('__')[0]} Coefficients across splits"
                )
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.show()
        else:
            raise TypeError(f"Unexpected report type: {type(self._parent.reports_[0])}")
