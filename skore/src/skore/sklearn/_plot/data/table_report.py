import itertools
import json
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from skrub import _column_associations
from skrub import _dataframe as sbd
from skrub._reporting._html import to_html
from skrub._reporting._summarize import summarize_dataframe
from skrub._reporting._utils import (
    JSONEncoder,
    duration_to_numeric,
    ellide_string,
    format_number,
    format_percent,
    top_k_value_counts,
)

from skore.sklearn._plot.style import StyleDisplayMixin
from skore.sklearn._plot.utils import (
    HelpDisplayMixin,
    ReprHTMLMixin,
    _adjust_fig_size,
    _rotate_ticklabels,
    _validate_style_kwargs,
)
from skore.skrub import _skrub_compat as sbd_compat

_RED = "tab:red"
_ORANGE = "tab:orange"
_TEXT_COLOR_PLACEHOLDER = "#123456"


def _get_range(values, frac=0.2, factor=3.0):
    min_value, low_p, high_p, max_value = np.quantile(
        values, [0.0, frac, 1.0 - frac, 1.0]
    )
    delta = high_p - low_p
    if not delta:
        return min_value, max_value
    margin = factor * delta
    low = low_p - margin
    high = high_p + margin

    # Chosen low bound should be max(low, min_value). Moreover, we add a small
    # tolerance: if the clipping value is close to the actual minimum, extend
    # it (so we don't clip right above the minimum which looks a bit silly).
    if low - margin * 0.15 < min_value:
        low = min_value
    if max_value < high + margin * 0.15:
        high = max_value
    return low, high


def _robust_hist(values, ax):
    low, high = _get_range(values)
    inliers = values[(low <= values) & (values <= high)]
    n_low_outliers = (values < low).sum()
    n_high_outliers = (high < values).sum()
    n, bins, patches = ax.hist(inliers)
    n_out = n_low_outliers + n_high_outliers
    if not n_out:
        return 0, 0
    width = bins[1] - bins[0]
    start, stop = bins[0], bins[-1]
    line_params = dict(color=_RED, linestyle="--", ymax=0.95)
    if n_low_outliers:
        start = bins[0] - width
        ax.stairs([n_low_outliers], [start, bins[0]], color=_RED, fill=True)
        ax.axvline(bins[0], **line_params)
    if n_high_outliers:
        stop = bins[-1] + width
        ax.stairs([n_high_outliers], [bins[-1], stop], color=_RED, fill=True)
        ax.axvline(bins[-1], **line_params)
    ax.text(
        0.33,
        0.90,
        (f"{format_number(n_out)} outliers ({format_percent(n_out / len(values))})"),
        transform=ax.transAxes,
        ha="left",
        va="baseline",
        fontweight="bold",
        color=_RED,
    )
    ax.set_xlim(start, stop)
    return n_low_outliers, n_high_outliers


def _has_no_decimal(df):
    for col in sbd.to_column_list(df):
        if sbd.any(col != sbd.cast(col, "int")):
            return False
    return True


def _global_max(df):
    return max([sbd.max(col) for col in sbd.to_column_list(df)])


def _truncate_top_k(col, k, other_label="other"):
    """Truncate a column to the top k most frequent values.

    Replaces the rest with a label 'other'.
    """
    if col is None or sbd.is_numeric(col):
        return col

    # Use only the top k most frequent items of the color column
    # if it's categorical.
    _, counter = top_k_value_counts(col, k=k)
    values, _ = zip(*counter, strict=False)
    other = sbd.make_column_like(col, [other_label] * sbd.shape(col)[0], name="c")
    keep = sbd_compat.is_in(col, values) | sbd.is_null(
        col
    )  # we don't want to replace NaN with 'other'
    col = sbd.where(col, keep, other)
    col = sbd.make_column_like(
        col,
        [ellide_string(s, max_len=20) for s in sbd.to_list(col)],
        name=sbd.name(col),
    )

    return col


def _mask_top_k(cols, names, k):
    """Select the top k most frequent pairs of values in cols.

    First select the top k occurrences, then fill the missing pairs.
    """
    key = "_skore_count"  # an arbitrary column name that disappear after pivoting.
    indices = (
        pd.DataFrame(cols)
        .T.assign(**{key: 1})
        .groupby(names)[key]
        .sum()
        .sort_values(ascending=False)
        .head(k)
        .index
    )
    left, right = zip(*list(indices), strict=False)
    left, right = list(set(left)), list(set(right))
    return list(itertools.product(left, right))


def _pairwise_product_index(x, y):
    """Create a list of all pairs of unique values in x and y."""
    return list(itertools.product(x.unique().tolist(), y.unique().tolist()))


def _aggregate_pairwise(x, y, hue, k, heatmap_kwargs):
    """Create a symmetric matrix by a pairwise aggregation of its columns.

    - If the color column hue is provided, the values of the symmetric matrix are
      the mean of hue for a given pair of (x, y) entries
    - Otherwise, the values of the symmetric matrix are the frequency of each pair
      (x, y).
    """
    # We use Pandas for pivot and groupby operations to simplify the logic.
    cols = [sbd.to_pandas(x), sbd.to_pandas(y)]
    names = [col.name for col in cols]
    mask_top_k = _mask_top_k(cols, names, k)
    full_index = _pairwise_product_index(*cols)
    if hue is None:
        key = "_skore_count"  # an arbitrary column name that disappear after pivoting.
        df = (
            pd.DataFrame(cols)
            .T.assign(**{key: 1})
            .groupby(names)[key]
            .sum()
            .reindex(full_index)
            .fillna(0)
            .loc[mask_top_k]
            .reset_index()
            .pivot(
                columns=names[0],
                index=names[1],
                values=key,
            )
        )
        cbar_kws = {"label": "total"}
    else:
        hue = sbd.to_pandas(hue)
        cols.append(hue)
        key = hue.name
        df = (
            pd.DataFrame(cols)
            .T.groupby(names)[key]
            .mean()
            .reindex(full_index)
            .loc[mask_top_k]
            .reset_index()
            .pivot(columns=names[0], index=names[1], values=key)
        )
        cbar_kws = {"label": f"average {ellide_string(key)}"}

    user_heatmap_kwargs = heatmap_kwargs.get("cbar_kws", {})
    for k, v in cbar_kws.items():
        user_heatmap_kwargs.setdefault(k, v)
    heatmap_kwargs["cbar_kws"] = user_heatmap_kwargs

    return {"df": df, "heatmap_kwargs": heatmap_kwargs}


def _require_x(x, kind):
    if x is None:
        raise ValueError(f"When {kind=!r}, ``x`` is mandatory.")


def _check_no_args(x, y, hue, kind):
    params = dict(
        x=x,
        y=y,
        hue=hue,
    )
    for k, v in params.items():
        if v is not None:
            raise ValueError(f"When {kind=!r}, this function takes no {k!r} argument.")


class TableReportDisplay(StyleDisplayMixin, HelpDisplayMixin, ReprHTMLMixin):
    """Display reporting information about a given dataset.

    This display summarizes the dataset and provides a way to visualize
    the distribution of its columns.

    Parameters
    ----------
    summary : dict
        The summary of the dataset, as returned by ``summarize_dataframe``.

    dataset : DataFrame
        The original dataset that was summarized.

    column_filters : dict[str, Any] | None, default=None
        Names of the column to keep in the display of the TableReport. If None,
        all columns are kept.
    """

    _default_heatmap_kwargs: dict[str, Any] | None = None
    _default_boxplot_kwargs: dict[str, Any] | None = None
    _default_scatterplot_kwargs: dict[str, Any] | None = None
    _default_stripplot_kwargs: dict[str, Any] | None = None

    def __init__(self, summary, dataset, column_filters=None):
        self.summary = summary
        self.column_filters = column_filters
        self.dataset = dataset

    @classmethod
    def _compute_data_for_display(cls, dataset, with_plots=True, title=None):
        summary = summarize_dataframe(
            dataset,
            with_plots=with_plots,
            title=title,
        )
        return cls(summary, dataset)

    @StyleDisplayMixin.style_plot
    def plot(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        kind: str = "dist",
        top_k_categories: int = 20,
        scatterplot_kwargs: dict[str, Any] | None = None,
        stripplot_kwargs: dict[str, Any] | None = None,
        boxplot_kwargs: dict[str, Any] | None = None,
        heatmap_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Plot a 1d or 2d distribution of the column(s) from the dataset.

        Parameters
        ----------
        x : str, default=None
            The name of the column to use for the x-axis of the plot. Mandatory when
            ``kind='dist'``.

        y : str, default=None
            The name of the column to use for the y-axis of the plot. Only used when
            ``kind='dist'``.

        hue : str, default=None
            The name of the column to use for the color or hue axis of the plot. Only
            used when ``kind='dist'``.

        kind : {'dist', 'corr'}, default='dist'
            The kind of plot drawn.

            - If ``'dist'``, plot a distribution parametrized by ``x``, ``y``
              and ``hue``. When only ``x`` is defined, the distribution is 1d.
              When ``y`` is also defined, the plot is the 2d. Finally, when the
              color is set using ``hue``, the distribution is 2d, with a color per
              data-point based on ``hue``. This mode handle both numeric and
              categorical columns.
            - If ``'corr'``, plot
              `Cramer's V <https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V>`_
              correlation among all columns. This option doesn't take any ``x``,
              ``y`` or ``hue`` argument.

        top_k_categories : int, default=20
            For categorical columns, the number of most frequent elements to display.
            Only used when ``kind='dist'``.

        scatterplot_kwargs: dict, default=None
            Keyword arguments to be passed to seaborn's ``scatterplot`` for rendering
            the distribution 2D plot, when both ``x`` and ``y`` are numeric.

        stripplot_kwargs: dict, default=None
            Keyword arguments to be passed to seaborn's ``stripplot`` for rendering
            the distribution 2D plot, when either ``x`` or ``y`` is numeric, and
            the other is categorical. This plot is drawn on top of the boxplot.

        boxplot_kwargs: dict, default=None
            Keyword arguments to be passed to seaborn's ``boxplot`` for rendering
            the distribution 2D plot, when either ``x`` or ``y`` is numeric, and
            the other is categorical. This plot is drawn below the stripplot.

        heatmap_kwargs: dict, default=None
            Keyword arguments to be passed to seaborn's ``heatmap`` for rendering
            the Cramer's V correlation matrix, when ``kind='corr'`` or when
            ``kind='dist'`` and both ``x`` and ``y`` are categorical.
        """
        self.fig_, self.ax_ = plt.subplots(dpi=150)
        if kind == "dist":
            _require_x(x, kind)
            self._plot_distribution(
                x=x,
                y=y,
                hue=hue,
                k=top_k_categories,
                scatterplot_kwargs=scatterplot_kwargs,
                stripplot_kwargs=stripplot_kwargs,
                boxplot_kwargs=boxplot_kwargs,
                heatmap_kwargs=heatmap_kwargs,
            )

        elif kind == "corr":
            _check_no_args(x, y, hue, kind)
            self._plot_cramer(heatmap_kwargs=heatmap_kwargs)

        else:
            raise ValueError(f"'kind' options are 'dist', 'corr', got {kind!r}.")

    def _plot_distribution(
        self,
        x,
        y=None,
        hue=None,
        k=20,
        scatterplot_kwargs=None,
        stripplot_kwargs=None,
        boxplot_kwargs=None,
        heatmap_kwargs=None,
    ):
        if y is None and hue is None:
            # XXX: should we allow 1d plot with a hue value (hue)?
            self._plot_distribution_1d(x=x, k=k)
        else:
            self._plot_distribution_2d(
                x=x,
                y=y,
                hue=hue,
                k=k,
                scatterplot_kwargs=scatterplot_kwargs,
                stripplot_kwargs=stripplot_kwargs,
                boxplot_kwargs=boxplot_kwargs,
                heatmap_kwargs=heatmap_kwargs,
            )

    def _plot_distribution_1d(self, *, x, k):
        col = sbd.col(self.dataset, x)

        duration_unit = None
        if sbd.is_duration(col):
            col, duration_unit = duration_to_numeric(col)

        if sbd.is_numeric(col) or sbd.is_any_date(col):
            self._histogram(col, duration_unit)
        else:
            _, value_counts = top_k_value_counts(col, k=k)
            self._value_counts(value_counts, n_rows=sbd.shape(col)[0])
        self.ax_.set_xlabel(x)
        self.ax_.set_ylabel("Total")

    def _histogram(self, col, duration_unit=None):
        """Histogram for a numeric column."""
        # XXX: adapt to use hist_kwargs and seaborn histplot?
        col = sbd.drop_nulls(col)
        if sbd.is_float(col):
            # avoid any issues with pandas nullable dtypes
            # (to_numpy can yield a numpy array with object dtype in old pandas
            # version if there are inf or nan)
            col = sbd.to_float32(col)
        values = sbd.to_numpy(col)
        _robust_hist(values, self.ax_)
        if duration_unit is not None:
            self.ax_.set_xlabel(f"{duration_unit.capitalize()}s")
        if sbd.is_any_date(col):
            _rotate_ticklabels(self.ax_)
        _adjust_fig_size(self.fig_, self.ax_, 6.0, 3.0)

    def _value_counts(self, value_counts, n_rows, color=_ORANGE):
        """Bar plot of the frequencies of the most frequent values in a column.

        Parameters
        ----------
        value_counts : list
            Pairs of (value, count). Must be sorted from most to least frequent.

        n_unique : int
            Cardinality of the plotted column, used to determine if all unique
            values are plotted or if there are too many and some have been
            omitted. The figure's title is adjusted accordingly.

        n_rows : int
            Total length of the column, used to convert the counts to proportions.

        color : str
            The color for the bars.

        Returns
        -------
        str
            The plot as a XML string.
        """
        # XXX: adapt to use value_counts_kwargs and seaborn barplot?
        values = [ellide_string(v) for v, _ in value_counts][::-1]
        counts = [c for _, c in value_counts][::-1]
        rects = self.ax_.barh(list(map(str, range(len(values)))), counts, color=color)
        percent = [format_percent(c / n_rows) for c in counts]
        large_percent = [
            f"{p: >6}" if c > counts[-1] / 2 else ""
            for (p, c) in zip(percent, counts, strict=False)
        ]
        small_percent = [
            p if c <= counts[-1] / 2 else ""
            for (p, c) in zip(percent, counts, strict=False)
        ]

        # those are written on top of the orange bars so we write them in black
        self.ax_.bar_label(
            rects, large_percent, padding=-30, color="black", fontsize=14
        )
        # those are written on top of the background so we write them in foreground
        # color
        self.ax_.bar_label(
            rects, small_percent, padding=5, color=_TEXT_COLOR_PLACEHOLDER, fontsize=14
        )

        self.ax_.set_yticks(self.ax_.get_yticks())
        self.ax_.set_yticklabels(list(map(str, values)))

        _adjust_fig_size(self.fig_, self.ax_, 7.0, 0.4 * len(values))

    def _plot_distribution_2d(
        self,
        *,
        x,
        y=None,
        hue=None,
        k=20,
        heatmap_kwargs,
        stripplot_kwargs,
        boxplot_kwargs,
        scatterplot_kwargs,
    ):
        x, y, hue = (
            sbd.col(self.dataset, x),
            sbd.col(self.dataset, y) if y is not None else None,
            sbd.col(self.dataset, hue) if hue is not None else None,
        )
        if y is None:
            y = hue

        is_x_num, is_y_num = sbd.is_numeric(x), sbd.is_numeric(y)
        if is_x_num and is_y_num:
            if scatterplot_kwargs is None:
                scatterplot_kwargs = self._default_scatterplot_kwargs or {}
            self._scatterplot(
                x=x,
                y=y,
                hue=_truncate_top_k(hue, k),
                scatterplot_kwargs=scatterplot_kwargs,
            )
        elif is_x_num or is_y_num:
            if is_y_num:
                x, y = y, x

            # We use a horizontal box plot to limit xlabels overlap.
            if boxplot_kwargs is None:
                boxplot_kwargs = self._default_boxplot_kwargs or {}
            if stripplot_kwargs is None:
                stripplot_kwargs = self._default_stripplot_kwargs or {}
            self._boxplot(
                x=x,
                y=_truncate_top_k(y, k),
                hue=_truncate_top_k(hue, k),
                stripplot_kwargs=stripplot_kwargs,
                boxplot_kwargs=boxplot_kwargs,
            )
        else:
            if (hue is not None) and (not sbd.is_numeric(hue)):
                raise ValueError(
                    "If 'x' and 'y' are categories, 'hue' must be continuous."
                )
            if heatmap_kwargs is None:
                heatmap_kwargs = self._default_heatmap_kwargs or {}
            self._heatmap(
                **_aggregate_pairwise(x, y, hue, k, heatmap_kwargs=heatmap_kwargs),
            )
        self.ax_.set_xlabel(sbd.name(x))
        self.ax_.set_ylabel(sbd.name(y))

    def _scatterplot(self, *, x, y, hue, scatterplot_kwargs):
        scatterplot_kwargs_validated = _validate_style_kwargs(
            {"alpha": 0.1},
            scatterplot_kwargs,
        )
        sns.scatterplot(x=x, y=y, hue=hue, ax=self.ax_, **scatterplot_kwargs_validated)
        if self.ax_.legend_ is not None:
            sns.move_legend(self.ax_, (1.05, 0.0))

    def _boxplot(self, *, x, y, hue, stripplot_kwargs, boxplot_kwargs):
        h = max(sbd.n_unique(y) * 0.3, 3)
        w = 6

        stripplot_kwargs_validated = _validate_style_kwargs(
            {
                "dodge": False,
                "size": 8,
                "alpha": 0.5,
                "zorder": 0,
            },
            stripplot_kwargs,
        )
        sns.stripplot(
            x=x,
            y=y,
            hue=hue,
            ax=self.ax_,
            **stripplot_kwargs_validated,
        )

        boxplot_kwargs_validated = _validate_style_kwargs(
            {
                "fliersize": 0,
                "width": 0.5,
                "whis": (0, 100),  # spellchecker:disable-line
                "linecolor": "#000000",
                "linewidth": 1.0,
                "color": "white",
                "boxprops": dict(alpha=0.1),
                "zorder": 1,
            },
            boxplot_kwargs,
        )
        sns.boxplot(
            x=x,
            y=y,
            ax=self.ax_,
            **boxplot_kwargs_validated,
        )

        _adjust_fig_size(self.fig_, self.ax_, w, h)

        if self.ax_.legend_ is not None:
            sns.move_legend(self.ax_, (1.05, 0.0))

    def _heatmap(self, df, *, title=None, heatmap_kwargs):
        df.index = [ellide_string(s) for s in df.index]
        df.columns = [ellide_string(s) for s in df.columns]

        n_rows, n_cols = sbd.shape(df)
        h = min(max(n_rows * 0.9, 4), 8)
        w = min(max(n_cols * 0.9, 4), 8)
        _adjust_fig_size(self.fig_, self.ax_, w, h)

        df = df.infer_objects(copy=False).fillna(np.nan)

        if _global_max(df) < 1000:  # noqa: SIM108
            # avoid scientific notation for small numbers
            fmt = ".0f" if _has_no_decimal(df) else ".2f"
        else:
            # scientific notation for bigger numbers
            fmt = ".2g"

        annot = n_cols < 10

        heatmap_kwargs_validated = _validate_style_kwargs(
            {
                "xticklabels": True,
                "yticklabels": True,
                "robust": True,
                "annot": annot,
                "fmt": fmt,
            },
            heatmap_kwargs,
        )

        sns.heatmap(
            df,
            ax=self.ax_,
            **heatmap_kwargs_validated,
        )
        if title is not None:
            self.ax_.set_title(title)

    def _plot_cramer(self, *, heatmap_kwargs: dict[str, Any] | None):
        """Plot Cramer's V correlation among all columns."""
        if heatmap_kwargs is None:
            heatmap_kwargs = self._default_heatmap_kwargs or {}

        cramer_v_table = pd.DataFrame(
            _column_associations._cramer_v_matrix(self.dataset),
            columns=self.dataset.columns,
            index=self.dataset.columns,
        )
        return self._heatmap(
            cramer_v_table,
            title="Cramer's V Correlation",
            heatmap_kwargs=heatmap_kwargs,
        )

    def frame(self):
        return self.summary

    def _html_snippet(self):
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

    def _html_repr(self, include=None, exclude=None):
        return self._html_snippet()

    def __repr__(self):
        return f"<{self.__class__.__name__}: use .open() to display>"

    def _json(self):
        to_remove = ["dataframe"]
        data = {k: v for k, v in self.summary.items() if k not in to_remove}
        return json.dumps(data, cls=JSONEncoder)
