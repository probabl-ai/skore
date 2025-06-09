import matplotlib.pyplot as plt

from skore.sklearn._plot.style import StyleDisplayMixin
from skore.sklearn._plot.utils import HelpDisplayMixin
from skore.sklearn.types import ReportType


class MetricsSummaryDisplay(HelpDisplayMixin, StyleDisplayMixin):
    """Display for summarize.

    An instance of this class will be created by `Report.metrics.summarize()`.
    This class should not be instantiated directly.
    """

    def __init__(
        self,
        *,
        summarize_data,
        report_type: ReportType,
    ):
        self.summarize_data = summarize_data
        self.report_type = report_type

    def frame(self):
        """Return the summarize as a dataframe."""
        return self.summarize_data

    @StyleDisplayMixin.style_plot
    def plot(self, x, y) -> None:
        """Plot visualization.

        Extra keyword arguments will be passed to matplotlib's `plot`.

        Parameters
        ----------
        x : str, default=None
            The metric to display on x-axis. By default, the first column.

        y : str, default=None
            The metric to display on y-axis. By default, the second column.

        Notes
        -----
        The average precision (cf. :func:`~sklearn.metrics.average_precision_score`)
        in scikit-learn is computed without any interpolation. To be consistent
        with this metric, the precision-recall curve is plotted without any
        interpolation as well (step-wise style).

        You can change this style by passing the keyword argument
        `drawstyle="default"`. However, the curve will not be strictly
        consistent with the reported average precision.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(classifier, **split_data)
        >>> display = report.metrics.precision_recall()
        >>> display.plot(pr_curve_kwargs={"color": "tab:red"})
        """
        self.figure_, self.ax_ = plt.subplots()

        if self.report_type in (
            ["estimator", "cross-validation", "comparison-cross-validation"]
        ):
            raise NotImplementedError("To come soon!")
        elif self.report_type == "comparison-estimator":
            self.plot_comparison_estimator()

    def plot_comparison_estimator(self):
        self.report_metrics_data.scatter(x=0, y=1)
