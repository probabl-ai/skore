import matplotlib.pyplot as plt

from skore.sklearn._plot.base import Display
from skore.sklearn._plot.style import StyleDisplayMixin
from skore.sklearn.utils import _SCORE_OR_LOSS_INFO


class PairPlotDisplay(Display):
    """Display for pair plot.

    Parameters
    ----------
    scatter_data :

    x_column : str

    y_column : str

    display_label_x : str, default=None

    display_label_y : str, default=None

    data_source : str, default=None


    Attributes
    ----------
    figure_ : matplotlib Figure
        Figure containing the confusion matrix.

    ax_ : matplotlib Axes
        Axes with confusion matrix.

    text_ : ndarray of shape (n_classes, n_classes), dtype=matplotlib Text or \
            None
        Array of matplotlib text elements containing the values in the
        confusion matrix.
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
        display_labels=None,
    ):
        self.scatter_data = scatter_data
        self.x_column = scatter_data.columns[0]
        self.y_column = scatter_data.columns[1]
        self.display_label_x = display_label_x
        self.display_label_y = display_label_y
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
            Configured with the confusion matrix.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        scatter_data = self.scatter_data

        ax.scatter(
            x=scatter_data[self.x_column],
            y=scatter_data[self.y_column],
            title=f"{self.display_label_x} vs {self.display_label_x} on \
                {self.data_source} data",
        )
        ax.set_xlabel(self.display_label_x)
        ax.set_ylabel(self.display_label_y)

        # Add labels to the points with a small offset
        text = scatter_data["Estimator"]
        x = scatter_data[self.x_column]
        y = scatter_data[self.y_column]
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

        self.figure_, self.ax_ = fig, ax
        return self

    @classmethod
    def from_metrics(
        cls,
        metrics,
        perf_metric_x,
        perf_metric_y,
        data_source,
    ):
        """Create a confusion matrix display from predictions.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True labels.

        y_pred : array-like of shape (n_samples,)
            Predicted labels, as returned by a classifier.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        display_labels : list of str, default=None
            Target names used for plotting. By default, labels will be inferred
            from y_true.

        include_values : bool, default=True
            Includes values in confusion matrix.

        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, confusion matrix will not be
            normalized.

        values_format : str, default=None
            Format specification for values in confusion matrix. If None, the format
            specification is 'd' or '.2g' whichever is shorter.

        Returns
        -------
        display : :class:`PairPlotDisplay`
            The confusion matrix display.
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
            x_column=None,
            y_column=None,
            display_label_x=x_label_text,
            display_label_y=y_label_text,
            data_source=None,
        )

        return disp

    def frame(self):
        """Return the confusion matrix as a dataframe.

        Returns
        -------
        scatter_data : pandas.DataFrame
            The dataframe used to create the scatter plot.
        """
        return self.scatter_data
