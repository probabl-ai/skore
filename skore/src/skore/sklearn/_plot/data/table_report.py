from skrub._reporting._html import to_html
from skrub._reporting._summarize import summarize_dataframe

from skore.sklearn._plot.base import Display


class TableReportDisplay(Display):
    def __init__(self, summary, column_filters=None):
        self.summary = summary
        self.column_filters = column_filters

    @classmethod
    def from_frame(cls, df, with_plots=True, title=None):
        summary = summarize_dataframe(
            df,
            with_plots=with_plots,
            title=title,
        )
        return cls(summary)

    def plot(self, kind=None):
        if kind is None:
            return self.html_snippet()
        # TODO
        raise NotImplementedError()

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

    def _repr_mimebundle_(self, include=None, exclude=None):
        del include, exclude
        return {"text/html": self.html_snippet()}

    def _repr_html_(self):
        return self._repr_mimebundle_()["text/html"]
