import pandas as pd

from skore._sklearn._plot.base import Display
from skore._sklearn._plot.style import StyleDisplayMixin
from skore._sklearn._plot.utils import HelpDisplayMixin, PlotBackendMixin


class FeatureImportanceDisplay(
    HelpDisplayMixin, StyleDisplayMixin, PlotBackendMixin, Display
):
    def __init__(self, coefficient_data, _parent):
        self.coefficient_data = coefficient_data
        self._parent = _parent

    def frame(self):
        """Return the coefficients as a dataframe."""
        if isinstance(self.coefficient_data, list):
            return pd.concat(self.coefficient_data, axis=1)
        return self.coefficient_data

    @StyleDisplayMixin.style_plot
    def _plot_matplotlib(self, **kwargs):
        import matplotlib.pyplot as plt

        # avoid circular imports
        from skore._sklearn._comparison.report import ComparisonReport
        from skore._sklearn._cross_validation.report import CrossValidationReport
        from skore._sklearn._estimator.report import EstimatorReport

        if isinstance(self._parent, EstimatorReport):
            self.coefficient_data.plot.bar()
        elif isinstance(self._parent, CrossValidationReport):
            plt.figure(figsize=(12, 6))
            self.coefficient_data.boxplot()
            plt.title("Coefficient variance across CV splits")
            plt.tight_layout()
            plt.show()
        elif isinstance(self._parent, ComparisonReport):
            if isinstance(self._parent.reports_[0], EstimatorReport):
                for coef_frame in self.coefficient_data:
                    _, ax = plt.subplots()
                    coef_frame.plot(
                        kind="bar",
                        ax=ax,
                        title=f"{coef_frame.columns[0]} Coefficients",
                    )
                    plt.tight_layout()
                    plt.show()
            elif isinstance(self._parent.reports_[0], CrossValidationReport):
                for coef_frame in self.coefficient_data:
                    _, ax = plt.subplots()
                    coef_frame.boxplot(ax=ax)
                    plt.xticks(rotation=90)
                    plt.tight_layout()
                    plt.show()
            else:
                raise TypeError(
                    f"Unexpected report type: {type(self._parent.reports_[0])}"
                )
        else:
            raise NotImplementedError(
                f"Cannot use plot() method on {type(self._parent)}"
            )
