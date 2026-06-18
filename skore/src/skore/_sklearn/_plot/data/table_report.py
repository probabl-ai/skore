from typing import Any, Literal

import narwhals as nw
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
    duration_to_numeric,
    ellide_string,
    top_k_value_counts,
)

from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn._plot.utils import (
    _adjust_fig_size,
    _rotate_ticklabels,
    _validate_style_kwargs,
)
from skore._utils._dataframe import UserDataFrame
from skore._utils.repr import ReprHTMLMixin


def _truncate_top_k_categories(
    col: pd.Series | None, k: int, other_label: str = "other"
) -> pd.Series:
    """Truncate a column to the top k most frequent values.

    Replaces the rest with a label defined by ``other_label``. Note that if `col` is not
    a categorical column, it is returned as is.

    Parameters
    ----------
    col : pd.Series or None
        The column to truncate.

    k : int
        The number of most frequent values to keep.

    other_label : str, default="other"
        The label to use for the rest of the values.

    Returns
    -------
    pd.Series
        The truncated column.
    """
    if col is None or pd.api.types.is_numeric_dtype(col):
        return col

    col = col.copy()
    _, counter = top_k_value_counts(col, k=k)
    values, _ = zip(*counter, strict=False)
    # we don't want to replace NaN with 'other'
    keep = col.isin(values) | col.isna()
    if isinstance(col.dtype, pd.CategoricalDtype):
        col = col.cat.add_categories(other_label)
        col[~keep] = other_label
        col = col.cat.remove_unused_categories()
        col = col.cat.rename_categories(
            {v: ellide_string(v, max_len=20) for v in values if isinstance(v, str)}
        )
    else:
        col[~keep] = other_label
        col = col.apply(
            lambda x: ellide_string(x, max_len=20) if isinstance(x, str) else x
        )
    return col


def _compute_contingency_table(
    x: pd.Series, y: pd.Series, hue: pd.Series | None, k: int
) -> pd.DataFrame:
    """Compute the contingency table of x and y with filtering only to the top k pairs.

    The contingency table is a symmetric matrix where the values are the mean of hue
    for a given pair of (x, y) entries. If hue is not provided, the values are the
    frequency of each pair (x, y).

    Parameters
    ----------
    x : pd.Series
        The first column to aggregate.

    y : pd.Series
        The second column to aggregate.

    hue : pd.Series | None
        The column to aggregate by.

    k : int
        The number of top pairs to select.

    Returns
    -------
    pd.DataFrame
        The contingency table.
    """
    if x.name is None or y.name is None:
        raise ValueError("The series x and y must have a name.")

    if hue is None:
        contingency_table = pd.crosstab(index=y, columns=x)
    else:
        contingency_table = pd.crosstab(index=y, columns=x, values=hue, aggfunc="mean")

    cols = pd.concat([x.to_frame(), y.to_frame()], axis=1)
    top_pairs = cols.value_counts().nlargest(k).index

    top_x_values = sorted({pair[0] for pair in top_pairs})
    top_y_values = sorted({pair[1] for pair in top_pairs})

    # As stated by a pandas warning, we call explicitly `infer_objects` to downcast
    # the contingency table and silence the warning using the context manager.
    with pd.option_context("future.no_silent_downcasting", True):
        return (
            contingency_table.fillna(0)
            .infer_objects(copy=False)
            .reindex(index=top_y_values, columns=top_x_values, fill_value=0)
        )


def _resize_categorical_axis(
    *,
    figure: Figure,
    ax: Axes,
    n_categories: int,
    is_x_axis: bool,
    size_per_category: float = 0.3,
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

    size_per_category : float, default=0.3
        The size of the category in the axis. When having annotations, the size needs
        to be tweaked based on the formatting of the annotations.
    """
    if is_x_axis:
        ax.tick_params(axis="x", length=0)
        target_height = figure.get_size_inches()[1]
        target_width = n_categories * size_per_category
    else:
        ax.tick_params(axis="y", length=0)
        target_height = n_categories * size_per_category
        target_width = figure.get_size_inches()[0]
    _adjust_fig_size(figure, ax, target_width, target_height)


class TableReportDisplay(ReprHTMLMixin, DisplayMixin):
    """Summarize and plot dataset columns.

    Parameters
    ----------
    summary : dict
        The summary of the dataset, as returned by ``summarize_dataframe``.

    Attributes
    ----------
    summary : dict
        Dataset summary produced by skrub's ``summarize_dataframe``.

    See Also
    --------
    EstimatorReport.data.summarize : Create this display from a report.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skore import evaluate
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> classifier = LogisticRegression(max_iter=10_000)
    >>> report = evaluate(classifier, X, y, splitter=0.2)
    >>> display = report.data.summarize()
    >>> display.plot(kind="corr")
    """

    _default_heatmap_kwargs: dict[str, Any] = {}
    _default_boxplot_kwargs: dict[str, Any] = {}
    _default_scatterplot_kwargs: dict[str, Any] = {}
    _default_stripplot_kwargs: dict[str, Any] = {}
    _default_histplot_kwargs: dict[str, Any] = {}

    def __init__(self, summary: dict[str, Any]) -> None:
        self.summary = summary

    @classmethod
    def _compute_data_for_display(cls, dataset: Any) -> "TableReportDisplay":
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
        with plt.ioff():
            return cls(
                summarize_dataframe(
                    dataset,
                    with_plots=True,
                    title=None,
                    verbose=0,
                )
            )

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        kind: Literal["dist", "corr"] = "dist",
        top_k_categories: int = 20,
    ) -> Figure:
        """Plot distribution or correlation of the columns from the dataset.

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

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the plot.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=0.2)
        >>> display = report.data.summarize()
        >>> display.plot(kind="corr")
        """
        return self._plot(
            x=x,
            y=y,
            hue=hue,
            kind=kind,
            top_k_categories=top_k_categories,
        )

    def _plot_matplotlib(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        kind: Literal["dist", "corr"] = "dist",
        top_k_categories: int = 20,
    ) -> Figure:
        """Matplotlib implementation of the `plot` method."""
        figure, ax = plt.subplots()
        if kind == "dist":
            match (x is None, y is None, hue is None):
                case (True, True, True) | (True, True, False):
                    raise ValueError(
                        "When kind='dist', at least one of x, y must be provided and "
                        "optionally hue. Got x=None, y=None."
                    )
                case (False, True, True) | (True, False, True):
                    self._plot_distribution_1d(
                        figure=figure,
                        ax=ax,
                        x=x,
                        y=y,
                        k=top_k_categories,
                        histplot_kwargs=self._default_histplot_kwargs,
                    )
                case _:
                    self._plot_distribution_2d(
                        figure=figure,
                        ax=ax,
                        x=x,
                        y=y,
                        hue=hue,
                        k=top_k_categories,
                        scatterplot_kwargs=self._default_scatterplot_kwargs,
                        stripplot_kwargs=self._default_stripplot_kwargs,
                        boxplot_kwargs=self._default_boxplot_kwargs,
                        heatmap_kwargs=self._default_heatmap_kwargs,
                    )

        elif kind == "corr":
            for param_name, param_value in zip(
                ("x", "y", "hue"), (x, y, hue), strict=True
            ):
                if param_value is not None:
                    raise ValueError(
                        f"When {kind=!r}, {param_name!r} argument must be None."
                    )
            self._plot_cramer(
                figure=figure, ax=ax, heatmap_kwargs=self._default_heatmap_kwargs
            )

        else:
            raise ValueError(f"'kind' options are 'dist', 'corr', got {kind!r}.")

        return figure

    def _plot_distribution_1d(
        self,
        *,
        figure: Figure,
        ax: Axes,
        x: str | None,
        y: str | None,
        k: int,
        histplot_kwargs: dict[str, Any],
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

        histplot_kwargs : dict
            Keyword arguments to be passed to :ref:`histplot
            <https://seaborn.pydata.org/generated/seaborn.histplot.html>`_ for rendering
            the distribution 1D plot.
        """
        default_histplot_kwargs: dict[str, Any] = {}

        column = nw.from_native(self.summary["dataframe"])[x or y]
        dtype = column.dtype

        duration_unit = None
        is_categorical = False
        if dtype == nw.Duration:
            # `duration_to_numeric` is a skrub helper operating on pandas.
            plot_column, duration_unit = duration_to_numeric(column.to_pandas())
        elif is_categorical := not (dtype.is_numeric() or dtype.is_temporal()):
            top_k = (
                column.drop_nulls().value_counts(sort=True).head(k)[column.name]
            ).to_list()
            # the ordered categorical encoding required by seaborn is pandas-only.
            plot_column = (
                column.filter(column.is_in(top_k))
                .to_pandas()
                .astype(pd.CategoricalDtype(categories=top_k, ordered=True))
            )
            default_histplot_kwargs["color"] = "tab:orange"
            default_histplot_kwargs["discrete"] = True
        else:
            plot_column = column.to_native()
            if dtype.is_integer():
                default_histplot_kwargs["discrete"] = True

        histplot_kwargs_validated = _validate_style_kwargs(
            default_histplot_kwargs, histplot_kwargs
        )

        if x is not None:
            histplot_params = {"x": plot_column}
            despine_params = {"bottom": is_categorical}
            if duration_unit is not None:
                ax.set(xlabel=f"{duration_unit.capitalize()}s")
        else:  # y is not None
            histplot_params = {"y": plot_column}
            despine_params = {"left": is_categorical}
            if duration_unit is not None:
                ax.set(ylabel=f"{duration_unit.capitalize()}s")

        sns.histplot(ax=ax, **histplot_params, **histplot_kwargs_validated)
        sns.despine(
            figure,
            top=True,
            right=True,
            trim=True,
            offset=10,
            **despine_params,
        )

        if is_categorical:
            _resize_categorical_axis(
                figure=figure,
                ax=ax,
                n_categories=plot_column.nunique(),
                is_x_axis=x is not None,
            )

        if x is not None and any(
            len(label.get_text()) > 1 for label in ax.get_xticklabels()
        ):
            # rotate only for string longer than 1 character
            _rotate_ticklabels(ax, rotation=45)

    def _plot_distribution_2d(
        self,
        *,
        figure: Figure,
        ax: Axes,
        x: str | None,
        y: str | None,
        heatmap_kwargs: dict[str, Any],
        stripplot_kwargs: dict[str, Any],
        boxplot_kwargs: dict[str, Any],
        scatterplot_kwargs: dict[str, Any],
        hue: str | None = None,
        k: int = 20,
    ) -> None:
        """Plot 2-dimensional distribution of two features.

        This function plots a 2-dimensional distribution of two features, with optional
        hue and k-most frequent categories.

        Parameters
        ----------
        x : str or None
            The name of the column to use for the x-axis of the plot.

        y : str or None
            The name of the column to use for the y-axis of the plot.

        heatmap_kwargs : dict
            Keyword arguments to be passed to heatmap.

        stripplot_kwargs : dict
            Keyword arguments to be passed to stripplot.

        boxplot_kwargs : dict
            Keyword arguments to be passed to boxplot.

        scatterplot_kwargs : dict
            Keyword arguments to be passed to scatterplot.

        hue : str, default=None
            The name of the column to use for the color or hue axis of the plot.

        k : int, default=20
            The number of most frequent categories to plot.
        """
        dataframe = nw.from_native(self.summary["dataframe"])
        x_col = dataframe[x] if x is not None else None
        y_col = dataframe[y] if y is not None else None
        hue_col = dataframe[hue] if hue is not None else None
        x_col, y_col = (
            hue_col if x_col is None else x_col,
            y_col if y_col is not None else hue_col,
        )

        despine_params = {"top": True, "right": True, "trim": True, "offset": 10}
        is_x_num = x_col is not None and x_col.dtype.is_numeric()
        is_y_num = y_col is not None and y_col.dtype.is_numeric()
        is_hue_num = hue_col is not None and hue_col.dtype.is_numeric()

        # `_truncate_top_k_categories`, `pd.crosstab` and the categorical encoding
        # below are pandas-only, and seaborn requires a single backend per call, so
        # materialize the vectors to pandas here.
        x_series: pd.Series | None = x_col.to_pandas() if x_col is not None else None
        y_series: pd.Series | None = y_col.to_pandas() if y_col is not None else None
        hue_series: pd.Series | None = (
            hue_col.to_pandas() if hue_col is not None else None
        )
        hue_series = _truncate_top_k_categories(hue_series, k)

        if is_x_num and is_y_num:
            scatterplot_kwargs_validated = _validate_style_kwargs(
                {"alpha": 0.1}, scatterplot_kwargs
            )
            sns.scatterplot(
                x=x_series,
                y=y_series,
                hue=hue_series,
                ax=ax,
                **scatterplot_kwargs_validated,
            )
        elif is_x_num or is_y_num:
            stripplot_kwargs_validated = _validate_style_kwargs(
                {
                    "dodge": False,
                    "size": 6,
                    "alpha": 0.5,
                    "zorder": 0,
                },
                stripplot_kwargs,
            )
            boxplot_kwargs_validated = _validate_style_kwargs(
                {
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
                },
                boxplot_kwargs,
            )

            if is_x_num:
                y_series = _truncate_top_k_categories(y_series, k)
            else:
                x_series = _truncate_top_k_categories(x_series, k)

            sns.boxplot(x=x_series, y=y_series, ax=ax, **boxplot_kwargs_validated)
            sns.stripplot(
                x=x_series,
                y=y_series,
                hue=hue_series,
                ax=ax,
                **stripplot_kwargs_validated,
            )

            categorical_series = y_series if is_x_num else x_series
            assert categorical_series is not None

            _resize_categorical_axis(
                figure=figure,
                ax=ax,
                n_categories=categorical_series.nunique(),
                is_x_axis=not is_x_num,
            )
            if is_x_num:
                despine_params["left"] = True
            else:
                despine_params["bottom"] = True
                if any(len(label.get_text()) > 1 for label in ax.get_xticklabels()):
                    _rotate_ticklabels(ax, rotation=45)
        else:
            if (hue_series is not None) and (not is_hue_num):
                raise ValueError(
                    "If 'x' and 'y' are categories, 'hue' must be continuous."
                )

            contingency_table = _compute_contingency_table(
                x_series, y_series, hue_series, k
            )
            contingency_table.index = [
                ellide_string(s) for s in contingency_table.index
            ]
            contingency_table.columns = [
                ellide_string(s) for s in contingency_table.columns
            ]

            if max_value := contingency_table.max(axis=None) < 100_000:
                # avoid scientific notation for small numbers
                annotation_format = (
                    ".0f"
                    if all(
                        pd.api.types.is_integer_dtype(dtype)
                        for dtype in contingency_table.dtypes
                    )
                    else ".2f"
                )
                size_per_category = len(str(max_value)) * 0.15
            else:
                # scientific notation for bigger numbers
                annotation_format = ".2g"
                size_per_category = 0.6

            cbar_kwargs = (
                {"label": f"Mean {hue_series.name}"}
                if hue_series is not None
                else {"label": "Count"}
            )

            heatmap_kwargs_validated = _validate_style_kwargs(
                {
                    "xticklabels": True,
                    "yticklabels": True,
                    "robust": True,
                    "annot": True,
                    "fmt": annotation_format,
                    "cbar_kws": cbar_kwargs,
                },
                heatmap_kwargs,
            )
            sns.heatmap(contingency_table, ax=ax, **heatmap_kwargs_validated)
            despine_params.update(left=True, bottom=True)
            ax.tick_params(axis="both", length=0)

            for is_x_axis, x_or_y in zip(
                [True, False],
                [
                    pd.Series(contingency_table.columns),
                    pd.Series(contingency_table.index),
                ],
                strict=False,
            ):
                _resize_categorical_axis(
                    figure=figure,
                    ax=ax,
                    n_categories=x_or_y.nunique(),
                    is_x_axis=is_x_axis,
                    size_per_category=size_per_category,
                )

        sns.despine(figure, **despine_params)

        ax.set(
            xlabel=x_series.name if x_series is not None else None,
            ylabel=y_series.name if y_series is not None else None,
        )
        if ax.legend_ is not None:
            sns.move_legend(ax, (1.05, 0.0))

    def _plot_cramer(
        self, *, figure: Figure, ax: Axes, heatmap_kwargs: dict[str, Any]
    ) -> None:
        """Plot Cramer's V correlation among all columns.

        Parameters
        ----------
        heatmap_kwargs : dict, default=None
            Keyword arguments to be passed to heatmap.
        """
        heatmap_kwargs_validated = _validate_style_kwargs(
            {
                "xticklabels": True,
                "yticklabels": True,
                "robust": True,
                "square": True,
                "annot": True,
                "annot_kws": {"size": 10},
                "fmt": ".2f",
            },
            heatmap_kwargs,
        )

        dataframe = self.summary["dataframe"]
        columns = nw.from_native(dataframe).columns

        cramer_v_table = pd.DataFrame(
            _column_associations._cramer_v_matrix(dataframe),
            columns=columns,
            index=columns,
        )
        # only show the lower triangle of the heatmap since it is a correlation matrix
        # and keep the diagonal as well.
        mask = np.triu(np.ones_like(cramer_v_table, dtype=bool), k=1)

        sns.heatmap(cramer_v_table, mask=mask, ax=ax, **heatmap_kwargs_validated)
        ax.set(title="Cramer's V Correlation")

    def frame(
        self, *, kind: Literal["dataset", "top-associations"] = "dataset"
    ) -> UserDataFrame:
        """Get the data related to the table report.

        Parameters
        ----------
        kind : {'dataset', 'top-associations'}
            The kind of data to return.

        Returns
        -------
        DataFrame
            The dataset used to create the table report. When ``kind="dataset"``,
            the dataframe is returned in the user's native backend (pandas or
            polars).
        """
        if kind == "dataset":
            return self.summary["dataframe"]
        elif kind == "top-associations":
            return pd.DataFrame(self.summary["top_associations"])
        else:
            raise ValueError(f"Invalid kind: {kind!r}")

    def _html_repr(self) -> str:
        """Show the HTML representation of the report."""
        return to_html(self.summary, standalone=False, column_filters=None)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(...)>"

    # ignore the type signature because we override kwargs by specifying the name of
    # the parameters for the user.
    def set_style(  # type: ignore[override]
        self,
        *,
        policy: Literal["override", "update"] = "update",
        scatterplot_kwargs: dict[str, Any] | None = None,
        stripplot_kwargs: dict[str, Any] | None = None,
        boxplot_kwargs: dict[str, Any] | None = None,
        heatmap_kwargs: dict[str, Any] | None = None,
        histplot_kwargs: dict[str, Any] | None = None,
    ):
        """Set the style parameters for the display.

        Parameters
        ----------
        policy : Literal["override", "update"], default="update"
            Policy to use when setting the style parameters.
            If "override", existing settings are set to the provided values.
            If "update", existing settings are not changed; only settings that were
            previously unset are changed.

        scatterplot_kwargs: dict, default=None
            Keyword arguments to be passed to :func:`seaborn.scatterplot` for
            rendering the distribution 2D plot, when both ``x`` and ``y`` are numeric.

        stripplot_kwargs: dict, default=None
            Keyword arguments to be passed to :func:`seaborn.stripplot` for
            rendering the distribution 2D plot, when either ``x`` or ``y`` is numeric,
            and the other is categorical. This plot is drawn on top of the boxplot.

        boxplot_kwargs: dict, default=None
            Keyword arguments to be passed to :func:`seaborn.boxplot` for
            rendering the distribution 2D plot, when either ``x`` or ``y`` is numeric,
            and the other is categorical. This plot is drawn below the stripplot.

        heatmap_kwargs: dict, default=None
            Keyword arguments to be passed to :func:`seaborn.heatmap` for
            rendering Cramer's V correlation matrix, when ``kind='corr'`` or when
            ``kind='dist'`` and both ``x`` and ``y`` are categorical.

        histplot_kwargs: dict, default=None
            Keyword arguments to be passed to :func:`seaborn.histplot` for
            rendering the distribution 1D plot, when only ``x`` is provided.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a style parameter is unknown.
        """
        return super().set_style(
            policy=policy,
            scatterplot_kwargs=scatterplot_kwargs or {},
            stripplot_kwargs=stripplot_kwargs or {},
            boxplot_kwargs=boxplot_kwargs or {},
            heatmap_kwargs=heatmap_kwargs or {},
            histplot_kwargs=histplot_kwargs or {},
        )
