import weakref

from skrub import _dataframe as sbd
from skrub._reporting._html import to_html
from skrub._reporting._serve import open_in_browser
from skrub._reporting._summarize import summarize_dataframe

from skore.sklearn._plot.data import _plotting
from skore.sklearn._plot.style import StyleDisplayMixin
from skore.sklearn._plot.utils import HelpDisplayMixin, ReprHTMLMixin


def _subsample(df, n_subsample, subsample_strategy, random_state):
    if n_subsample is None:
        return df
    if subsample_strategy == "head":
        return sbd.head(df, n_subsample)
    elif subsample_strategy == "random":
        return sbd.sample(df, n_subsample, seed=random_state)
    else:
        raise ValueError(
            "'subsample_strategy' options are 'head', 'random', "
            f"got {subsample_strategy}."
        )


class TableReportDisplay(StyleDisplayMixin, HelpDisplayMixin, ReprHTMLMixin):
    def __init__(self, summary, df, column_filters=None):
        self.summary = summary
        self.column_filters = column_filters
        self._df = weakref.ref(df)

    @property
    def df(self):
        if (_df := self._df()) is None:
            raise ValueError("The dataset is not accessible by the report anymore.")
        return _df

    @classmethod
    def _compute_data_for_display(cls, df, with_plots=True, title=None):
        summary = summarize_dataframe(
            df,
            with_plots=with_plots,
            title=title,
        )
        return cls(summary, df)

    @StyleDisplayMixin.style_plot
    def plot_dist(
        self,
        x_col,
        y_col=None,
        c_col=None,
        n_subsample=None,
        subsample_strategy="head",
        random_state=None,
    ):
        """Plot a 1d or 2d distribution of the column(s) from the dataset.

        Parameters
        ----------
        x_col : str
            The name of the column to use for the x-axis of the plot.

        y_col : str, default=None
            The name of the column to use for the y-axis of the plot.

        c_col : str, default=None
            The name of the column to use for the color or hue axis of the plot.

        Return
        ------
        fig : matplotlib.figure.Figure
            The drawn figure.
        """
        return _plotting.plot_distribution(
            _subsample(self.df, n_subsample, subsample_strategy, random_state),
            x_col,
            y_col,
            c_col,
        )

    @StyleDisplayMixin.style_plot
    def plot_pearson(self):
        return _plotting.plot_pearson(self.df)

    @StyleDisplayMixin.style_plot
    def plot_cramer(self):
        return _plotting.plot_cramer(self.df)

    def frame(self):
        return self.summary

    def html_snippet(self):
        """Get the report as an HTML fragment that can be inserted in a page.

        Returns
        -------
        str :
            The HTML snippet.
        """
        return to_html(
            self.summary,
            standalone=False,
            column_filters=self.column_filters,
        )

    def html(self):
        """Get the report as a full HTML page.

        Returns
        -------
        str :
            The HTML page.
        """
        return to_html(
            self.summary,
            standalone=True,
            column_filters=self.column_filters,
        )

    def _html_repr(self, include=None, exclude=None):
        return self.html_snippet()

    def __repr__(self):
        return f"<{self.__class__.__name__}: use .open() to display>"

    def open(self):
        """Open the HTML report in a web browser."""
        open_in_browser(self.html())
