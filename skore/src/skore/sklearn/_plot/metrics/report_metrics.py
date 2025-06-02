from skore.sklearn._plot.style import StyleDisplayMixin
from skore.sklearn._plot.utils import HelpDisplayMixin


class ReportMetricsDisplay(HelpDisplayMixin, StyleDisplayMixin):
    """Display for report_metrics.

    An instance of this class is should created by `Report.metrics.report_metrics()`.
    You should not create an instance of this class directly.
    """

    def __init__(
        self,
        report_metrics_data,
    ):
        self.report_metrics_data = report_metrics_data

    def frame(self):
        """Return the confusion matrix as a dataframe.

        Returns
        -------
        frame : pandas.DataFrame
            The confusion matrix as a dataframe.
        """
        return self.report_metrics_data
