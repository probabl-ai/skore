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


def _require_x_col(x_col, kind):
    if x_col is None:
        raise ValueError(f"When {kind=!r}, ``x_col`` is mandatory.")


def _check_no_args(x_col, y_col, c_col, kind):
    params = dict(
        x_col=x_col,
        y_col=y_col,
        c_col=c_col,
    )
    for k, v in params.items():
        if v is not None:
            raise ValueError(f"When {kind=!r}, this function takes no {k!r} argument.")


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
    def plot(
        self,
        x_col=None,
        y_col=None,
        c_col=None,
        kind="dist",
        n_subsample=None,
        subsample_strategy="head",
        random_state=None,
    ):
        """Plot a 1d or 2d distribution of the column(s) from the dataset.

        Parameters
        ----------
        x_col : str, default=None
            The name of the column to use for the x-axis of the plot.

        y_col : str, default=None
            The name of the column to use for the y-axis of the plot.

        c_col : str, default=None
            The name of the column to use for the color or hue axis of the plot.

        kind : {'dist', 'pearson', 'cramer'}, default='dist'
            The kind of plot drawn.

            - If ``'dist'``, plot a distribution parametrized by ``x_col``, ``y_col``
              and ``c_col``. When only ``x_col`` is defined, the distribution is 1d.
              When ``y_col`` is also defined, the plot is the 2d. Finally, when the
              color is set using ``c_col``, the distribution is 2d, with a color per
              data-point based on ``c_col``. This mode handle both numeric and
              categorical columns.
            - If ``'pearson'``, plot Pearson's correlation among numeric columns. This
              option doesn't take any ``x_col``, ``y_col`` or ``c_col`` argument.
            - If ``'cramer'``, plot Cramer's V correlation among all columns. This
              option doesn't take any ``x_col``, ``y_col`` or ``c_col`` argument.

        n_subsample : int, default=None
            The number of points to subsample the dataframe hold by the display, using
            the strategy set by ``subsample_strategy``. It must be a strictly positive
            integer. If ``None``, no subsampling is applied.

        subsample_strategy : {'head', 'random'}, default='head',
            The strategy used to subsample the dataframe hold by the display. It only
            has an effect when ``n_subsample`` is not None.

            - If ``'head'``: subsample by taking the ``n_subsample`` first points of the
              dataframe, similar to Pandas: ``df.head(n)``.
            - If ``'random'``: randomly subsample the dataframe by using a uniform
              distribution. The random seed is controlled by ``random_state``.

        random_state : int, default=None
            The random seed to use when randomly subsampling. It only has an effect when
            ``n_subsample`` is not Noen and ``subsample_strategy='random'``.

        Return
        ------
        fig : matplotlib.figure.Figure
            The drawn figure.
        """
        df = _subsample(self.df, n_subsample, subsample_strategy, random_state)
        if kind == "dist":
            _require_x_col(x_col, kind)
            return _plotting.plot_distribution(
                df,
                x_col,
                y_col,
                c_col,
            )

        elif kind == "pearson":
            _check_no_args(x_col, y_col, c_col, kind)
            return _plotting.plot_pearson(df)

        elif kind == "cramer":
            _check_no_args(x_col, y_col, c_col, kind)
            return _plotting.plot_cramer(df)

        else:
            raise ValueError(
                f"'kind' options are 'dist', 'pearson', 'cramer', got {kind!r}."
            )

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
