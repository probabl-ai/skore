import itertools
import json
from typing import Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from skrub import _column_associations
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
    _adjust_fig_size,
    _rotate_ticklabels,
    _validate_style_kwargs,
)
from skore.skrub._skrub_compat import sbd
from skore.utils._repr_html import ReprHTMLMixin


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


def _resize_categorical_axis(
    *,
    figure: Figure,
    ax: Axes,
    n_categories: int,
    is_x_axis: bool,
) -> None:
    """Resize the axis along which categories are plotted.

    Parameters
    ----------
    fig : Figure
        The figure to resize.

    ax : Axes
        The axis to resize.

    is_x_axis : bool
        Whether the axis is the x-axis.
    """
    if is_x_axis:
        ax.tick_params(axis="x", length=0)
        target_height = figure.get_size_inches()[1]
        target_width = n_categories * 0.3
    else:
        ax.tick_params(axis="y", length=0)
        target_height = n_categories * 0.3
        target_width = figure.get_size_inches()[0]
    _adjust_fig_size(figure, ax, target_width, target_height)


class TableReportDisplay(StyleDisplayMixin, HelpDisplayMixin, ReprHTMLMixin):
    """Display reporting information about a given dataset.

    This display summarizes the dataset and provides a way to visualize
    the distribution of its columns.

    Parameters
    ----------
    summary : dict
        The summary of the dataset, as returned by ``summarize_dataframe``.

    Attributes
    ----------
    ax_ : matplotlib axes
        The axes of the figure.

    figure_ : matplotlib figure.
        The figure of the plot.

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
    >>> display = report.data.analyze()
    >>> display.plot(kind="corr")
    """

    _default_heatmap_kwargs: dict[str, Any] | None = None
    _default_boxplot_kwargs: dict[str, Any] | None = None
    _default_scatterplot_kwargs: dict[str, Any] | None = None
    _default_stripplot_kwargs: dict[str, Any] | None = None
    _default_histplot_kwargs: dict[str, Any] | None = None

    def __init__(self, summary: dict[str, Any]) -> None:
        self.summary = summary

    @classmethod
    def _compute_data_for_display(cls, dataset: pd.DataFrame) -> "TableReportDisplay":
        """Private method to create a TableReportDisplay from a dataset.

        Parameters
        ----------
        dataset : DataFrame
            The dataset to summarize.

        Returns
        -------
        display : TableReportDisplay
            Object that stores computed values.
        """
        return cls(summarize_dataframe(dataset, with_plots=True, title=None, verbose=0))

    @StyleDisplayMixin.style_plot
    def plot(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        kind: Literal["dist", "corr"] = "dist",
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
        self.figure_, self.ax_ = plt.subplots()
        if kind == "dist":
            match (x is None, y is None, hue is None):
                case (True, True, True) | (True, True, False):
                    raise ValueError(
                        "When kind='dist', at least one of x, y must be provided and "
                        "optionally hue. Got x=None, y=None, hue=None."
                    )
                case (False, True, True) | (True, False, True):
                    if histplot_kwargs is None:
                        histplot_kwargs = self._default_histplot_kwargs

                    self._plot_distribution_1d(
                        x=x, y=y, k=top_k_categories, histplot_kwargs=histplot_kwargs
                    )
                case _:
                    if scatterplot_kwargs is None:
                        scatterplot_kwargs = self._default_scatterplot_kwargs
                    if stripplot_kwargs is None:
                        stripplot_kwargs = self._default_stripplot_kwargs
                    if boxplot_kwargs is None:
                        boxplot_kwargs = self._default_boxplot_kwargs
                    self._plot_distribution_2d(
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
            for param_name, param_value in zip(
                ("x", "y", "hue"), (x, y, hue), strict=True
            ):
                if param_value is not None:
                    raise ValueError(
                        f"When {kind=!r}, {param_name!r} argument should be None."
                    )
            if heatmap_kwargs is None:
                heatmap_kwargs = self._default_heatmap_kwargs
            self._plot_cramer(heatmap_kwargs=heatmap_kwargs)

        else:
            raise ValueError(f"'kind' options are 'dist', 'corr', got {kind!r}.")

    def _plot_distribution_1d(
        self,
        *,
        x: str | None,
        y: str | None,
        k: int,
        histplot_kwargs: dict[str, Any] | None,
    ) -> None:
        """Plot 1-dimensional distribution of a feature.

        Parameters
        ----------
        x : str, default=None
            The name of the column to plot.

        y : str, default=None
            The name of the column to plot.

        k : int
            The number of most frequent categories to plot.

        histplot_kwargs : dict, default=None
            Keyword arguments to be passed to seaborn's :ref:`histplot
            <https://seaborn.pydata.org/generated/seaborn.histplot.html>`_ for rendering
            the distribution 1D plot.
        """
        default_histplot_kwargs: dict[str, Any] = {}

        column = sbd.col(self.summary["dataframe"], x or y)

        duration_unit = None
        if sbd.is_duration(column):
            column, duration_unit = duration_to_numeric(column)

        if is_categorical := not (sbd.is_numeric(column) or sbd.is_any_date(column)):
            top_k = sbd.col(sbd.head(sbd.value_counts(column), k), "value")

            column = sbd.filter(column, sbd.is_in(column, top_k))
            column = sbd.to_pandas(column)
            column = column.astype(
                pd.CategoricalDtype(categories=sbd.to_list(top_k), ordered=True)
            )
            default_histplot_kwargs["color"] = "tab:orange"

        if sbd.is_integer(column) or is_categorical:
            default_histplot_kwargs["discrete"] = True

        histplot_kwargs_validated = _validate_style_kwargs(
            default_histplot_kwargs, histplot_kwargs or {}
        )

        if x is not None:
            histplot_params = dict(x=column)
            despine_params = dict(bottom=is_categorical)
            if duration_unit is not None:
                self.ax_.set(xlabel=f"{duration_unit.capitalize()}s")
        else:  # y is not None
            histplot_params = dict(y=column)
            despine_params = dict(left=is_categorical)
            if duration_unit is not None:
                self.ax_.set(ylabel=f"{duration_unit.capitalize()}s")

        sns.histplot(ax=self.ax_, **histplot_params, **histplot_kwargs_validated)
        sns.despine(
            self.figure_,
            top=True,
            right=True,
            trim=True,
            offset=10,
            **despine_params,
        )

        if is_categorical:
            _resize_categorical_axis(
                figure=self.figure_,
                ax=self.ax_,
                n_categories=sbd.n_unique(column),
                is_x_axis=x is not None,
            )

        if x is not None and any(
            len(label.get_text()) > 1 for label in self.ax_.get_xticklabels()
        ):
            # rotate only for string longer than 1 character
            _rotate_ticklabels(self.ax_, rotation=45)

    def _plot_distribution_2d(
        self,
        *,
        x: str | None,
        y: str | None,
        hue: str | None = None,
        k: int = 20,
        heatmap_kwargs: dict[str, Any] | None = None,
        stripplot_kwargs: dict[str, Any] | None = None,
        boxplot_kwargs: dict[str, Any] | None = None,
        scatterplot_kwargs: dict[str, Any] | None = None,
    ):
        """Plot 2-dimensional distribution of two features.

        This function plots a 2-dimensional distribution of two features, with optional
        hue and k-most frequent categories.

        Parameters
        ----------
        x : str or None
            The name of the column to use for the x-axis of the plot.

        y : str or None
            The name of the column to use for the y-axis of the plot.

        hue : str, default=None
            The name of the column to use for the color or hue axis of the plot.

        k : int, default=20
            The number of most frequent categories to plot.

        heatmap_kwargs : dict, default=None
            Keyword arguments to be passed to seaborn's heatmap.

        stripplot_kwargs : dict, default=None
            Keyword arguments to be passed to seaborn's stripplot.

        boxplot_kwargs : dict, default=None
            Keyword arguments to be passed to seaborn's boxplot.

        scatterplot_kwargs : dict, default=None
            Keyword arguments to be passed to seaborn's scatterplot.
        """
        default_scatterplot_kwargs: dict[str, Any] = {"alpha": 0.1}
        default_stripplot_kwargs: dict[str, Any] = {
            "dodge": False,
            "size": 6,
            "alpha": 0.5,
            "zorder": 0,
        }
        default_boxplot_kwargs: dict[str, Any] = {
            "fliersize": 0,
            "width": 0.5,
            "whis": (0, 100),  # spellchecker:disable-line
            "color": "black",
            "fill": False,
            "zorder": 1,
            "boxprops": {"alpha": 0.5},
            "whiskerprops": {"alpha": 0.5},
            "capprops": {"alpha": 0.5},
            "medianprops": {"alpha": 0.5},
        }
        default_heatmap_kwargs: dict[str, Any] = {}

        x = sbd.col(self.summary["dataframe"], x) if x is not None else None
        y = sbd.col(self.summary["dataframe"], y) if y is not None else None
        hue = sbd.col(self.summary["dataframe"], hue) if hue is not None else None
        x, y = hue if x is None else x, y if y is not None else hue

        despine_params = {}
        is_x_num, is_y_num = sbd.is_numeric(x), sbd.is_numeric(y)
        if is_x_num and is_y_num:
            scatterplot_kwargs_validated = _validate_style_kwargs(
                default_scatterplot_kwargs, scatterplot_kwargs or {}
            )
            sns.scatterplot(
                x=x,
                y=y,
                hue=_truncate_top_k(hue, k),
                ax=self.ax_,
                **scatterplot_kwargs_validated,
            )
        elif is_x_num or is_y_num:
            stripplot_kwargs_validated = _validate_style_kwargs(
                default_stripplot_kwargs, stripplot_kwargs or {}
            )
            boxplot_kwargs_validated = _validate_style_kwargs(
                default_boxplot_kwargs, boxplot_kwargs or {}
            )

            sns.stripplot(x=x, y=y, hue=hue, ax=self.ax_, **stripplot_kwargs_validated)
            sns.boxplot(x=x, y=y, ax=self.ax_, **boxplot_kwargs_validated)

            _resize_categorical_axis(
                figure=self.figure_,
                ax=self.ax_,
                n_categories=sbd.n_unique(y) if is_x_num else sbd.n_unique(x),
                is_x_axis=not is_x_num,
            )
            if is_x_num:
                despine_params["left"] = True
            else:
                despine_params["bottom"] = True
                if any(
                    len(label.get_text()) > 1 for label in self.ax_.get_xticklabels()
                ):
                    _rotate_ticklabels(self.ax_, rotation=45)
        else:
            if (hue is not None) and (not sbd.is_numeric(hue)):
                raise ValueError(
                    "If 'x' and 'y' are categories, 'hue' must be continuous."
                )
            heatmap_kwargs_validated = _validate_style_kwargs(
                default_heatmap_kwargs, heatmap_kwargs or {}
            )
            self._heatmap(
                **_aggregate_pairwise(
                    x, y, hue, k, heatmap_kwargs=heatmap_kwargs_validated
                ),
            )

        sns.despine(
            self.figure_,
            top=True,
            right=True,
            trim=True,
            offset=10,
            **despine_params,
        )

        self.ax_.set(xlabel=sbd.name(x), ylabel=sbd.name(y))
        if self.ax_.legend_ is not None:
            sns.move_legend(self.ax_, (1.05, 0.0))

    def _heatmap(self, df, *, title=None, heatmap_kwargs):
        df.index = [ellide_string(s) for s in df.index]
        df.columns = [ellide_string(s) for s in df.columns]

        n_rows, n_cols = sbd.shape(df)
        h = min(max(n_rows * 0.9, 4), 8)
        w = min(max(n_cols * 0.9, 4), 8)
        _adjust_fig_size(self.figure_, self.ax_, w, h)

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
            _column_associations._cramer_v_matrix(self.summary["dataframe"]),
            columns=self.summary["dataframe"].columns,
            index=self.summary["dataframe"].columns,
        )
        return self._heatmap(
            cramer_v_table,
            title="Cramer's V Correlation",
            heatmap_kwargs=heatmap_kwargs,
        )

    def frame(self, *, kind: Literal["dataset", "top-associations"] = "dataset"):
        """Get the data related to the table report.

        Parameters
        ----------
        kind : {'dataset', 'top-associations'}
            The kind of data to return.

        Returns
        -------
        DataFrame
            The dataset used to create the table report.
        """
        if kind == "dataset":
            return self.summary["dataframe"]
        elif kind == "top-associations":
            return pd.DataFrame(self.summary["top_associations"])
        else:
            return ValueError(f"Invalid kind: {kind!r}")

    def _html_repr(self) -> str:
        """Show the HTML representation of the report."""
        return to_html(self.summary, standalone=False, column_filters=None)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(...)>"

    def _json(self) -> str:
        """Serialize the data of this report in JSON format.

        It the serialization chosen to be sent to skore-hub.
        """
        to_remove = ["dataframe"]
        data = {k: v for k, v in self.summary.items() if k not in to_remove}
        return json.dumps(data, cls=JSONEncoder)
