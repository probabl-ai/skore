import matplotlib.pyplot as plt
import pandas as pd

from skore.sklearn._plot.style import StyleDisplayMixin
from skore.sklearn._plot.utils import HelpDisplayMixin
from skore.sklearn.types import ReportType


class MetricsSummaryDisplay(HelpDisplayMixin, StyleDisplayMixin):
    """Display for summarize.

    An instance of this class will be created by `Report.metrics.summarize()`.
    This class should not be instantiated directly.
    """

    # should be removed once transformed into a utils
    _SCORE_OR_LOSS_INFO: dict[str, dict[str, str]] = {
        "fit_time": {"name": "Fit time (s)", "icon": "(↘︎)"},
        "predict_time": {"name": "Predict time (s)", "icon": "(↘︎)"},
        "accuracy": {"name": "Accuracy", "icon": "(↗︎)"},
        "precision": {"name": "Precision", "icon": "(↗︎)"},
        "recall": {"name": "Recall", "icon": "(↗︎)"},
        "brier_score": {"name": "Brier score", "icon": "(↘︎)"},
        "roc_auc": {"name": "ROC AUC", "icon": "(↗︎)"},
        "log_loss": {"name": "Log loss", "icon": "(↘︎)"},
        "r2": {"name": "R²", "icon": "(↗︎)"},
        "rmse": {"name": "RMSE", "icon": "(↘︎)"},
        "custom_metric": {"name": "Custom metric", "icon": ""},
        "report_metrics": {"name": "Report metrics", "icon": ""},
    }

    def __init__(
        self, *, summarize_data, report_type: ReportType, data_source: str = "test"
    ):
        self.summarize_data = summarize_data
        self.report_type = report_type
        self.data_source = data_source

    def frame(self):
        """Return the summarize as a dataframe."""
        return self.summarize_data

    @StyleDisplayMixin.style_plot
    def plot(self, x, y) -> None:
        """Plot visualization.

        Parameters
        ----------
        x : str, default=None
            The metric to display on x-axis. By default, the first column.

        y : str, default=None
            The metric to display on y-axis. By default, the second column.

        Attributes
        ----------
        ax_ : matplotlib axes or ndarray of axes
            The axes on which the precision-recall curve is plotted.

        figure_ : matplotlib figure
            The figure on which the precision-recall curve is plotted.

        Returns
        -------
        A matplotlib plot.
        """
        if self.report_type in (
            ["estimator", "cross-validation", "comparison-cross-validation"]
        ):
            raise NotImplementedError("To come soon!")
        elif self.report_type == "comparison-estimator":
            self.plot_comparison_estimator(x, y)

    def plot_comparison_estimator(self, x, y):
        self.figure_, self.ax_ = plt.subplots()

        x_label = self._SCORE_OR_LOSS_INFO.get(x, {}).get("name", x)
        y_label = self._SCORE_OR_LOSS_INFO.get(y, {}).get("name", y)

        # Check that the metrics are in the report
        # If the metric is not in the report, help the user by suggesting
        # supported metrics
        reverse_score_info = {
            value["name"]: key for key, value in self._SCORE_OR_LOSS_INFO.items()
        }
        index = self.summarize_data.index
        if isinstance(index, pd.MultiIndex):
            available_columns = index.get_level_values(0).to_list()
        else:
            available_columns = index.tolist()
        supported_metrics = [
            reverse_score_info.get(col, col) for col in available_columns
        ]
        if x not in supported_metrics:
            raise ValueError(
                f"Performance metric {x} not found in the report. "
                f"Supported metrics are: {supported_metrics}."
            )
        if y not in supported_metrics:
            raise ValueError(
                f"Performance metric {y} not found in the report. "
                f"Supported metrics are: {supported_metrics}."
            )

        x_data = self.summarize_data.loc[x_label]
        y_data = self.summarize_data.loc[y_label]
        if len(x_data.shape) > 1 and x_data.shape[0] > 1:
            # case where we have multiIndex, and the metric is not a single value
            raise ValueError(
                "The perf metric x requires to add a positive label parameter."
            )
        elif len(x_data.shape) > 1 and x_data.shape[0] == 1:
            # case where we have multiIndex, but the metric is not affected by the
            # pos_label
            x_data = x_data.squeeze()
        if len(y_data.shape) > 1 and y_data.shape[0] > 1:
            raise ValueError(
                "The perf metric y requires to add a positive label parameter."
            )
        elif len(y_data.shape) > 1 and y_data.shape[0] == 1:
            y_data = y_data.squeeze()

        # Make it clear in the axis labels that we are using the train set
        if x == "fit_time" and self.data_source != "train":
            x_label_text = x_label + " on train set"
        else:
            x_label_text = x_label
        if y == "fit_time" and self.data_source != "train":
            y_label_text = y_label + " on train set"
        else:
            y_label_text = y_label

        title = f"{x_label} vs {y_label}"
        if self.data_source is not None:
            title += f" on {self.data_source} data"

        self.ax_.scatter(x=x_data, y=y_data)
        self.ax_.set_title(title)
        self.ax_.set_xlabel(x_label_text)
        self.ax_.set_ylabel(y_label_text)

        # Add labels to the points with a small offset
        text = self.summarize_data.columns.tolist()
        for label, x_coord, y_coord in zip(text, x_data, y_data, strict=False):
            self.ax_.annotate(
                label,
                (x_coord, y_coord),
                textcoords="offset points",
                xytext=(10, 0),
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    edgecolor="gray",
                    facecolor="white",
                    alpha=0.7,
                ),
            )

        plt.tight_layout()
        return self.figure_, self.ax_
