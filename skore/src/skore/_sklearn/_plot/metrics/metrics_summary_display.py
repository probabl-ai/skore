import itertools

import matplotlib.pyplot as plt
import pandas as pd

from skore._sklearn._plot.style import StyleDisplayMixin
from skore._sklearn._plot.utils import (
    HelpDisplayMixin,
    PlotBackendMixin,
    _interval_max_min_ratio,
)
from skore._sklearn.types import ReportType, ScoringName


class MetricsSummaryDisplay(HelpDisplayMixin, StyleDisplayMixin, PlotBackendMixin):
    """Display for summarize.

    An instance of this class will be created by `Report.metrics.summarize()`.
    This class should not be instantiated directly.
    """

    def __init__(
        self,
        *,
        summarize_data,
        report_type: ReportType,
        data_source: str = "test",
        default_verbose_metric_names: dict[str, dict[str, str]],
        scoring_names: ScoringName | list[ScoringName] | None = None,
    ):
        self.summarize_data = summarize_data
        self.report_type = report_type
        self.data_source = data_source
        self.scoring_names = scoring_names
        self.default_verbose_metric_names = default_verbose_metric_names

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
        if self.report_type in (
            ["estimator", "cross-validation", "comparison-cross-validation"]
        ):
            raise NotImplementedError("To come soon!")
        elif self.report_type == "comparison-estimator":
            self._plot_matplotlib_comparison_estimator(x, y)

    def _plot_matplotlib_comparison_estimator(self, x, y):
        _, ax = plt.subplots()

        # Get verbose name from x and y
        # if they are not verbose already
        x_verbose = self.default_verbose_metric_names.get(x, {}).get("name", x)
        y_verbose = self.default_verbose_metric_names.get(y, {}).get("name", y)

        # Check that the metrics are in the report
        # If the metric is not in the report, help the user by suggesting
        # supported metrics
        reverse_score_info = {
            value["name"]: key
            for key, value in self.default_verbose_metric_names.items()
        }
        available_metrics = self.summarize_data.index
        if isinstance(available_metrics, pd.MultiIndex):
            available_metrics = available_metrics.get_level_values(0).to_list()

        # if scoring_names is provided, they are the supported metrics
        # otherwise, the default verbose names apply.
        if self.scoring_names is not None:
            supported_metrics = self.scoring_names
        else:
            supported_metrics = [
                reverse_score_info.get(col, col) for col in available_metrics
            ]

        if x not in supported_metrics:
            raise ValueError(
                f"Performance metric '{x}' not found in the report. "
                f"Supported metrics are: {supported_metrics}."
            )
        if y not in supported_metrics:
            raise ValueError(
                f"Performance metric '{y}' not found in the report. "
                f"Supported metrics are: {supported_metrics}."
            )

        x_data = self.summarize_data.loc[x_verbose]
        y_data = self.summarize_data.loc[y_verbose]
        if len(x_data.shape) > 1:
            if x_data.shape[0] == 1:
                x_data = x_data.reset_index(drop=True).values[0]
            else:
                raise ValueError(
                    "The perf metric x requires to add a positive label parameter."
                )
        if len(y_data.shape) > 1:
            if y_data.shape[0] == 1:
                y_data = y_data.reset_index(drop=True).values[0]
            else:
                raise ValueError(
                    "The perf metric y requires to add a positive label parameter."
                )

        # Make it clear in the axis labels that we are using the train set
        if x == "fit_time" and self.data_source != "train":
            x_label_text = x_verbose + " on train set"
        else:
            x_label_text = x_verbose
        if y == "fit_time" and self.data_source != "train":
            y_label_text = y_verbose + " on train set"
        else:
            y_label_text = y_verbose

        title = f"{x_verbose} vs {y_verbose}"
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

        if _interval_max_min_ratio(x_data) > 5:
            xscale = "symlog" if x_data.min() <= 0 else "log"
        else:
            xscale = "linear"

        if _interval_max_min_ratio(y_data) > 5:
            yscale = "symlog" if y_data.min() <= 0 else "log"
        else:
            yscale = "linear"

        ax.set(
            title=title,
            xlabel=x_label_text,
            ylabel=y_label_text,
            xscale=xscale,
            yscale=yscale,
        )
        ax.legend(title="Models", loc="best")

        self.ax_ = ax
