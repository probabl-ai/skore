import pandas as pd

from skore._sklearn._plot.base import Display
from skore._sklearn._plot.style import StyleDisplayMixin
from skore._sklearn._plot.utils import PlotBackendMixin


class FeatureImportanceDisplay(StyleDisplayMixin, PlotBackendMixin, Display):
    def __init__(self, coefficient_data, _parent):
        self.coefficient_data = coefficient_data
        self._parent = _parent

    def frame(self):
        """Get the coefficients as a dataframe."""
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
            reshaped = self.coefficient_data.stack().reset_index()
            reshaped.columns = ["Feature", "Run", "Coefficient"]
            reshaped["Run"] = reshaped.groupby("Feature").cumcount()

            grouped = reshaped.groupby("Feature")["Coefficient"]
            data = [group.tolist() for _, group in grouped]
            plt.figure(figsize=(12, 6))
            plt.boxplot(data, labels=grouped.groups.keys())
            plt.xticks(rotation=45)
            plt.title("Coefficient variance across CV splits")
            plt.tight_layout()
            plt.show()
        elif isinstance(self._parent, ComparisonReport):
            if len(self.coefficient_data) == 1:
                self.coefficient_data[0].plot.bar()
                plt.show()
            else:
                for coef_frame in self.coefficient_data:
                    fix, ax = plt.subplots()
                    coef_frame.plot(
                        kind="bar", ax=ax, title=f"{coef_frame.columns[0]} Coefficients"
                    )
                    plt.tight_layout()
                    plt.show()
        else:
            raise NotImplementedError(
                f"Cannot use plot() method on {type(self._parent)}"
            )
