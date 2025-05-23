import matplotlib.pyplot as plt

from skore.sklearn._plot.base import Display
from skore.sklearn._plot.style import StyleDisplayMixin
from skore.sklearn.utils import _SCORE_OR_LOSS_INFO


class PairPlotDisplay(Display):
    """Display for pair plot.

    Parameters
    ----------
    scatter_data : pandas.DataFrame
        Dataframe containing the data to plot.

    x_column : str
        The name of the column to plot on the x-axis.
        If None, the first column of the dataframe is used.

    y_column : str
        The name of the column to plot on the y-axis.
        If None, the second column of the dataframe is used.

    display_label_x : str, default=None
        The label to use for the x-axis. If None, the name of the column will be used.

    display_label_y : str, default=None
        The label to use for the y-axis. If None, the name of the column will be used.

    data_source : str, default=None
        To specify the data source for the plot.

    Attributes
    ----------
    figure_ : matplotlib Figure
        Figure containing the pair plot.

    ax_ : matplotlib Axes
        Axes with pair plot.
    """

    @StyleDisplayMixin.style_plot
    def __init__(
        self,
        scatter_data,
        *,
        x_column=None,
        y_column=None,
        display_label_x=None,
        display_label_y=None,
        data_source=None,
    ):
        self.scatter_data = scatter_data
        if x_column is None:
            x_column = scatter_data.columns[0]
        self.x_column = x_column
        if y_column is None:
            y_column = scatter_data.columns[1]
        self.y_column = y_column
        self.display_label_x = (
            display_label_x if display_label_x is not None else self.x_column
        )
        self.display_label_y = (
            display_label_y if display_label_y is not None else self.y_column
        )
        self.data_source = data_source
        self.figure_ = None
        self.ax_ = None
        self.text_ = None

    def plot(self, ax=None, **kwargs):
        """Plot a given performance metric against another.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If None, a new figure and axes is created.

        **kwargs : dict
            Additional keyword arguments to be passed to matplotlib's
            `ax.imshow`.

        Returns
        -------
        self : PairPlotDisplay
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        scatter_data = self.scatter_data

        title = f"{self.display_label_x} vs {self.display_label_x}"
        if self.data_source is not None:
            title += f" on {self.data_source} data"
        ax.scatter(x=scatter_data[self.x_column], y=scatter_data[self.y_column])
        ax.set_title(title)
        ax.set_xlabel(self.display_label_x)
        ax.set_ylabel(self.display_label_y)

        self.figure_, self.ax_ = fig, ax
        return self

    @classmethod
    def from_metrics(
        cls,
        metrics,
        perf_metric_x,
        perf_metric_y,
        data_source=None,
    ):
        """Create a pair plot display from metrics.

        Parameters
        ----------
        metrics : pandas.DataFrame
            Dataframe containing the data to plot. The dataframe should
            contain the performance metrics for each estimator.

        perf_metric_x : str
            The name of the column to plot on the x-axis.

        perf_metric_y : str
            The name of the column to plot on the y-axis.

        data_source : str
            To specify the data source for the plot.

        Returns
        -------
        display : :class:`PairPlotDisplay`
            The scatter plot display.
        """
        x_label = _SCORE_OR_LOSS_INFO.get(perf_metric_x, {}).get("name", perf_metric_x)
        y_label = _SCORE_OR_LOSS_INFO.get(perf_metric_y, {}).get("name", perf_metric_y)
        scatter_data = metrics

        # Check that the metrics are in the report
        # If the metric is not in the report, help the user by suggesting
        # supported metrics
        reverse_score_info = {
            value["name"]: key for key, value in _SCORE_OR_LOSS_INFO.items()
        }
        available_columns = scatter_data.columns.get_level_values(0).to_list()
        available_columns.remove("Estimator")
        supported_metrics = [
            reverse_score_info.get(col, col) for col in available_columns
        ]
        if perf_metric_x not in supported_metrics:
            raise ValueError(
                f"Performance metric {perf_metric_x} not found in the report. "
                f"Supported metrics are: {supported_metrics}."
            )
        if perf_metric_y not in supported_metrics:
            raise ValueError(
                f"Performance metric {perf_metric_y} not found in the report. "
                f"Supported metrics are: {supported_metrics}."
            )

        # Check that x and y are 1D arrays (i.e. the metrics don't need pos_label)
        x = scatter_data[x_label]
        y = scatter_data[y_label]
        if len(x.shape) > 1:
            raise ValueError(
                "The perf metric x requires to add a positive label parameter."
            )
        if len(y.shape) > 1:
            raise ValueError(
                "The perf metric y requires to add a positive label parameter."
            )

        # Make it clear in the axis labels that we are using the train set
        if perf_metric_x == "fit_time" and data_source != "train":
            x_label_text = x_label + " on train set"
        else:
            x_label_text = x_label
        if perf_metric_y == "fit_time" and data_source != "train":
            y_label_text = y_label + " on train set"
        else:
            y_label_text = y_label

        disp = cls(
            scatter_data=scatter_data,
            x_column=x_label,
            y_column=y_label,
            display_label_x=x_label_text,
            display_label_y=y_label_text,
            data_source=data_source,
        ).plot()

        # Add labels to the points with a small offset
        ax = disp.ax_
        text = scatter_data["Estimator"]
        for label, x_coord, y_coord in zip(text, x, y):
            ax.annotate(
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

        disp.ax_ = ax
        return disp

    def frame(self):
        """Return the dataframe used for the pair plot.

        Returns
        -------
        scatter_data : pandas.DataFrame
            The dataframe used to create the scatter plot.
        """
        return self.scatter_data
