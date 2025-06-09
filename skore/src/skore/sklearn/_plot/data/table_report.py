from matplotlib import pyplot as plt
from skrub._reporting import _plotting
from skrub._reporting._html import to_html
from skrub._reporting._serve import open_in_browser
from skrub._reporting._summarize import summarize_dataframe

from skore.sklearn._plot.style import StyleDisplayMixin
from skore.sklearn._plot.utils import HelpDisplayMixin, ReprHTMLMixin


def distribution_1d(df, col_x, col_y=None):
    del col_y
    fig, ax = plt.subplots()
    _plotting._robust_hist(df[col_x], ax=ax, color=None)
    ax.set_title(col_x)


def distribution_2d(df, col_x, col_y=None):
    del col_y


def pearson(df, col_x, col_y=None):
    del col_y


def cramer(df, col_x, col_y=None):
    del col_y


class TableReportDisplay(StyleDisplayMixin, HelpDisplayMixin, ReprHTMLMixin):
    def __init__(self, summary, df, column_filters=None):
        self.summary = summary
        self.column_filters = column_filters
        # Use a weakref to store df?
        self.df = df

    @classmethod
    def _compute_data_for_display(cls, df, with_plots=True, title=None):
        summary = summarize_dataframe(
            df,
            with_plots=with_plots,
            title=title,
        )
        return cls(summary, df)

    @StyleDisplayMixin.style_plot
    def plot(self, kind, col_x, col_y=None):
        plots = {
            "distribution_1d": distribution_1d,
            "distribution_2d": distribution_2d,
            "pearson": pearson,
            "cramer": cramer,
        }
        if (func := plots.get(kind)) is None:
            raise ValueError(f"'kind' options are {list(plots)!r}, got {kind!r}.")

        return func(self.df, col_x, col_y)

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
