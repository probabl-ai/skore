import itertools
import json
from functools import partial
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

_ORANGE = "tab:orange"


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
    keep = sbd.is_in(col, values) | sbd.is_null(
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
    _default_histplot_kwargs: dict[str, Any] | None = None

    def __init__(self, summary, dataset, column_filters=None):
        self.summary = summary
        self.dataset = dataset
        self.column_filters = column_filters

    @classmethod
    def _compute_data_for_display(cls, dataset, with_plots=True, title=None):
        summary = summarize_dataframe(
            dataset,
            with_plots=with_plots,
            title=title,
            verbose=0,
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
        histplot_kwargs: dict[str, Any] | None = None,
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

            - If ``'dist'``, plot a distribution parametrized by ``x``, ``y`` and
              ``hue``. When only ``x`` is defined, the distribution is 1d. When ``y`` is
              also defined, the plot is the 2d. Finally, when the color is set using
              ``hue``, the distribution is 2d, with a color per data-point based on
              ``hue``. This mode handle both numeric and categorical columns.
            - If ``'corr'``, plot `Cramer's V
              <https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V>`_ correlation among all
              columns. This option doesn't take any ``x``, ``y`` or ``hue`` argument.

        top_k_categories : int, default=20
            For categorical columns, the number of most frequent elements to display.
            Only used when ``kind='dist'``.

        scatterplot_kwargs: dict, default=None
            Keyword arguments to be passed to seaborn's :ref:`scatterplot
            <https://seaborn.pydata.org/generated/seaborn.scatterplot.html>`_ for
            rendering the distribution 2D plot, when both ``x`` and ``y`` are numeric.

        stripplot_kwargs: dict, default=None
            Keyword arguments to be passed to seaborn's :ref:`stripplot
            <https://seaborn.pydata.org/generated/seaborn.stripplot.html>`_ for
            rendering the distribution 2D plot, when either ``x`` or ``y`` is numeric,
            and the other is categorical. This plot is drawn on top of the boxplot.

        boxplot_kwargs: dict, default=None
            Keyword arguments to be passed to seaborn's :ref:`boxplot
            <https://seaborn.pydata.org/generated/seaborn.boxplot.html>`_ for rendering
            the distribution 2D plot, when either ``x`` or ``y`` is numeric, and the
            other is categorical. This plot is drawn below the stripplot.

        heatmap_kwargs: dict, default=None
            Keyword arguments to be passed to seaborn's :ref:`heatmap
            <https://seaborn.pydata.org/generated/seaborn.heatmap.html>`_ for rendering
            Cramer's V correlation matrix, when ``kind='corr'`` or when ``kind='dist'``
            and both ``x`` and ``y`` are categorical.

        histplot_kwargs: dict, default=None
            Keyword arguments to be passed to seaborn's :ref:`histplot
            <https://seaborn.pydata.org/generated/seaborn.histplot.html>`_ for rendering
            the distribution 1D plot, when only ``x`` is provided.
        """
        self.fig_, self.ax_ = plt.subplots()
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
                histplot_kwargs=histplot_kwargs,
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
        histplot_kwargs=None,
    ):
        if y is None and hue is None:
            self._plot_distribution_1d(x=x, k=k, histplot_kwargs=histplot_kwargs)
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

    def _plot_distribution_1d(self, *, x, k, histplot_kwargs):
        x = sbd.col(self.dataset, x)

        duration_unit = None
        if sbd.is_duration(x):
            x, duration_unit = duration_to_numeric(x)

        if histplot_kwargs is None:
            histplot_kwargs = self._default_histplot_kwargs or {"despine": True}

        if is_object := not (sbd.is_numeric(x) or sbd.is_any_date(x)):
            top_k = sbd.col(sbd.head(sbd.value_counts(x), k), "value")

            x = sbd.filter(x, sbd.is_in(x, top_k))
            x = sbd.to_pandas(x)
            x = x.astype(
                pd.CategoricalDtype(categories=sbd.to_list(top_k), ordered=True)
            )
            histplot_kwargs["color"] = _ORANGE

        if sbd.is_integer(x) or is_object:
            histplot_kwargs["discrete"] = True

        despine = histplot_kwargs.pop("despine", True)
        histplot_kwargs_validated = _validate_style_kwargs(
            histplot_kwargs,
            {},
        )

        histplot = partial(sns.histplot, ax=self.ax_, **histplot_kwargs_validated)
        if is_object:
            histplot(y=x)
        else:
            histplot(x=x)

        if duration_unit is not None:
            self.ax_.set_xlabel(f"{duration_unit.capitalize()}s")
        if sbd.is_any_date(x):
            _rotate_ticklabels(self.ax_)
        if despine:
            # _despine_matplotlib_axis doesn't help.
            offset = -sbd.n_unique(x) + 10 if is_object else 0
            sns.despine(
                self.fig_,
                top=True,
                right=True,
                trim=is_object,
                offset={"bottom": offset},
            )

        if is_object:
            h = sbd.n_unique(x) * 0.3
            w = 6
            _adjust_fig_size(self.fig_, self.ax_, w, h)

    def _plot_distribution_2d(
        self,
        *,
        x,
        y,
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
