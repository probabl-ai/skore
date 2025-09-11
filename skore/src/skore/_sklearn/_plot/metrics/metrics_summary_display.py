from skore._sklearn._plot.base import DisplayMixin


class MetricsSummaryDisplay(DisplayMixin):
    """Display for summarize.

    An instance of this class will be created by `Report.metrics.summarize()`.
    This class should not be instantiated directly.
    """

    def __init__(self, summarize_data):
        self.summarize_data = summarize_data

    def frame(self):
        """Return the summarize as a dataframe.

        Returns
        -------
        frame : pandas.DataFrame
            The report metrics as a dataframe.
        """
        return self.summarize_data

    @DisplayMixin.style_plot
    def plot(self):
        """Not yet implemented."""
        raise NotImplementedError
