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
    parent : EstimatorReport | CrossValidationReport | ComparisonReport
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

        combined = pd.concat(
            [
                df.reset_index().reindex(
                    range(max(len(df) for df in self._coefficient_data))
                )
                for df in self._coefficient_data
            ],
            axis=1,
            keys=None,
        )
        # An 'index' column is a column with feature names
        feature_name_col_pos = [
            i
            for i, col in enumerate(combined.columns)
            if col in ("index", "Split index")
        ]
        if self._parent._reports_type == "EstimatorReport":
            if len(feature_name_col_pos) == 1:
                # If the two models use the same features, there is 1 'index' column
                combined.set_index("index", inplace=True)
                combined.index.name = None
            elif len(feature_name_col_pos) == 2:
                feature_name_col_1 = combined.iloc[:, feature_name_col_pos[0]].dropna()
                feature_name_col_2 = combined.iloc[:, feature_name_col_pos[1]].dropna()
                if set(feature_name_col_1).issubset(set(feature_name_col_2)) or set(
                    feature_name_col_2
                ).issubset(set(feature_name_col_1)):
                    # If one index is a subset of the other, set the output frame's
                    # index to be the full index.
                    # Otherwise, keep both feature ('index') columns
                    nan_counts = [
                        combined.iloc[:, pos].isna().sum()
                        for pos in feature_name_col_pos
                    ]
                    best_position = feature_name_col_pos[
                        nan_counts.index(min(nan_counts))
                    ]
                    combined.set_index(combined.iloc[:, best_position], inplace=True)
                    combined.drop(
                        combined.columns[feature_name_col_pos], axis=1, inplace=True
                    )
                    combined.index.name = None
        elif self._parent._reports_type == "CrossValidationReport":
            # Here the feature name columns just include unnecessary split index numbers
            combined.drop(combined.columns[feature_name_col_pos], axis=1, inplace=True)
            combined.index.name = "Split index"

        return combined

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
        if self._parent._reports_type == "EstimatorReport":
            for coef_frame in self._coefficient_data:
                self.figure_, self.ax_ = plt.subplots()
                coef_frame.plot.bar(ax=self.ax_)
                self.ax_.legend(loc="best", bbox_to_anchor=(1, 1), borderaxespad=1)

                self.ax_.set_title("Coefficients")
                self.figure_.tight_layout()
                plt.show()
        elif self._parent._reports_type == "CrossValidationReport":
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
