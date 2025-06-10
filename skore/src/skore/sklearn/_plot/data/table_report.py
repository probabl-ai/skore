import weakref

from skrub._reporting._html import to_html
from skrub._reporting._serve import open_in_browser
from skrub._reporting._summarize import summarize_dataframe

from skore.sklearn._plot.data import _plotting
from skore.sklearn._plot.style import StyleDisplayMixin
from skore.sklearn._plot.utils import HelpDisplayMixin, ReprHTMLMixin


class TableReportDisplay(StyleDisplayMixin, HelpDisplayMixin, ReprHTMLMixin):
    def __init__(self, summary, df, column_filters=None):
        self.summary = summary
        self.column_filters = column_filters
        # Use a weakref to store df?
        self._df = weakref.ref(df)

    @property
    def df(self):
        return self._df()

    @classmethod
    def _compute_data_for_display(cls, df, with_plots=True, title=None):
        summary = summarize_dataframe(
            df,
            with_plots=with_plots,
            title=title,
        )
        return cls(summary, df)

    @StyleDisplayMixin.style_plot
    def dist(self, *, x_col=None, y_col=None, c_col=None, kind="dist"):
        plots = {
            "dist": _plotting.plot_distribution,
            "pearson": _plotting.plot_pearson,
            "cramer": _plotting.plot_cramer,
        }
        if (func := plots.get(kind)) is None:
            raise ValueError(f"'kind' options are {list(plots)!r}, got {kind!r}.")

        return func(self.df, x_col, y_col, c_col)

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
