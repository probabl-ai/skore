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
        from skore._sklearn._comparison import ComparisonReport

        if isinstance(self._parent, ComparisonReport):
            return pd.concat(self.coefficient_data, axis=1)
        return self.coefficient_data

    @StyleDisplayMixin.style_plot
    def _plot_matplotlib(self, **kwargs):
        import matplotlib.pyplot as plt

        from skore._sklearn._comparison import ComparisonReport
        from skore._sklearn._cross_validation import CrossValidationReport
        from skore._sklearn._estimator import EstimatorReport

        if isinstance(self._parent, EstimatorReport):
            self.coefficient_data.plot.bar()
            plt.title(f"{self._parent.estimator_name_} Coefficients")
            plt.tight_layout()
            plt.show()
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
                    coef_frame.plot.bar(ax=ax)
                    if len(self.coefficient_data) == 1 or len(coef_frame.columns) > 1:
                        # If there's only one DataFrame, or if the plot includes
                        # multiple models with the same features, use a combined title
                        # like "Model 1 vs Model 2 Coefficients".
                        #
                        # For example, if 3 reports are passed and 2 share the same
                        # features, those two are plotted together with a combined
                        # title, while the third goes to the else clause and gets its
                        # title.
                        plt.title(f"{' vs '.join(coef_frame.columns)} Coefficients")
                    else:
                        plt.title(f"{coef_frame.columns[0]} Coefficients")
                    plt.tight_layout()
                    plt.show()
            elif isinstance(self._parent.reports_[0], CrossValidationReport):
                for coef_frame in self.coefficient_data:
                    _, ax = plt.subplots()
                    coef_frame.boxplot(ax=ax)
                    plt.title(
                        f"{coef_frame.columns[0].split('__')[0]} "
                        "Coefficients across splits"
                    )
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
