import itertools

import matplotlib.pyplot as plt
import pandas as pd

from skore._sklearn._plot.style import StyleDisplayMixin
from skore._sklearn._plot.utils import HelpDisplayMixin, PlotBackendMixin
from skore._sklearn.types import ReportType


class MetricsSummaryDisplay(HelpDisplayMixin, StyleDisplayMixin, PlotBackendMixin):
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
        """Return the summarize as a dataframe.

        Returns
        -------
        frame : pandas.DataFrame
            The report metrics as a dataframe.
        """
        return self.summarize_data

    def _plot_matplotlib(self, x: str, y: str) -> None:
        """Plot visualization.

        Parameters
        ----------
        x : str, default=None
            The metric to display on x-axis. By default, the first column.

        y : str, default=None
            The metric to display on y-axis. By default, the second column.
        """
        self.figure_, self.ax_ = plt.subplots()

        if self.report_type in (
            ["estimator", "cross-validation", "comparison-cross-validation"]
        ):
            raise NotImplementedError("To come soon!")
        elif self.report_type == "comparison-estimator":
            self._plot_comparison_estimator(x, y)

    def _plot_comparison_estimator(self, x, y):
        _, ax = plt.subplots()

        x_label = self._SCORE_OR_LOSS_INFO.get(x, {}).get("name", x)
        y_label = self._SCORE_OR_LOSS_INFO.get(y, {}).get("name", y)

        # Check that the metrics are in the report
        # If the metric is not in the report, help the user by suggesting
        # supported metrics
        reverse_score_info = {
            value["name"]: key for key, value in self._SCORE_OR_LOSS_INFO.items()
        }
        # available_columns = self.summarize_data.columns.get_level_values(0).to_list()
        # available_columns.remove("Estimator")
        available_columns = self.summarize_data.index
        if isinstance(available_columns, pd.MultiIndex):
            available_columns = available_columns.get_level_values(0).to_list()
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
        if len(x_data.shape) > 1:
            if x_data.shape[0] == 1:
                x_data = x_data.reset_index(drop=True).values
            else:
                raise ValueError(
                    "The perf metric x requires to add a positive label parameter."
                )
        if len(y_data.shape) > 1:
            if y_data.shape[0] == 1:
                y_data = y_data.reset_index(drop=True).values
            else:
                raise ValueError(
                    "The perf metric y requires to add a positive label parameter."
                )

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
            title += f" on {self.data_source} set"

        # Add legend
        text = self.summarize_data.columns
        markers = itertools.cycle(("o", "s", "^", "D", "v", "P", "*", "X", "h", "8"))
        colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

        for label, x_coord, y_coord in zip(text, x_data, y_data, strict=False):
            marker = next(markers)
            color = next(colors)
            ax.scatter(x_coord, y_coord, marker=marker, color=color, label=label)

        ax.set(title=title, xlabel=x_label_text, ylabel=y_label_text)
        ax.legend(title="Models", loc="best")

        self.ax_ = ax
