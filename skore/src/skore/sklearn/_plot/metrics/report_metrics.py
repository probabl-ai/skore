from skore.sklearn._plot.style import StyleDisplayMixin
from skore.sklearn._plot.utils import HelpDisplayMixin


class ReportMetricsDisplay(HelpDisplayMixin, StyleDisplayMixin):
    """Display for report_metrics.

    An instance of this class will be created by `Report.metrics.summarize()`.
    You should not create an instance of this class directly.
    """

    def __init__(self, report_metrics_data):
        self.report_metrics_data = report_metrics_data

    def frame(self):
        """Return the report metrics as a dataframe.

        Returns
        -------
        frame : pandas.DataFrame
            The report metrics as a dataframe.
        """
        return self.report_metrics_data
