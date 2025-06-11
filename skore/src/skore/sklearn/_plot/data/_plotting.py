import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from skrub import _column_associations
from skrub import _dataframe as sbd
from skrub._reporting import _utils

from skore.skrub._skrub_compat import is_in

_RED = "#dd0000"
_BLUE = "#4878d0"
_ORANGE = "#ee854a"
_TEXT_COLOR_PLACEHOLDER = "#123456"


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("none")


def _to_em(pt_match):
    attr, pt = pt_match.groups()
    pt = float(pt)
    px = pt * 96 / 72
    em = px / 16
    return f'{attr}="{em:.2f}em"'


def _rotate_ticklabels(ax):
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment="right")


def _get_adjusted_fig_size(fig, ax, direction, target_size):
    size_display = getattr(ax.get_window_extent(), direction)
    size = fig.dpi_scale_trans.inverted().transform((size_display, 0))[0]
    dim = 0 if direction == "width" else 1
    fig_size = fig.get_size_inches()[dim]
    return target_size * (fig_size / size)


def _adjust_fig_size(fig, ax, target_w, target_h):
    """Rescale a figure to the target width and height.

    This allows us to have all figures of a given type (bar plots, lines or
    histograms) have the same size, so that the displayed report looks more
    uniform, without having to do manual adjustments to account for the length
    of labels, occurrence of titles etc. We let pyplot generate the figure
    without any size constraints then resize it and thus let matplotlib deal
    with resizing the appropriate elements (eg shorter bars when the labels
    take more horizontal space).
    """
    w = _get_adjusted_fig_size(fig, ax, "width", target_w)
    h = _get_adjusted_fig_size(fig, ax, "height", target_h)
    fig.set_size_inches((w, h))


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
        # we place the text offset from the left rather than centering it to
        # make room for the factor matplotlib sometimes places on the right of
        # the axis eg "1e6" when the ticks are labelled in millions.
        0.15,
        1.0,
        (
            f"{_utils.format_number(n_out)} outliers "
            f"({_utils.format_percent(n_out / len(values))})"
        ),
        transform=ax.transAxes,
        ha="left",
        va="baseline",
        fontweight="bold",
        color=_RED,
    )
    ax.set_xlim(start, stop)
    return n_low_outliers, n_high_outliers


def histogram(col, duration_unit=None):
    """Histogram for a numeric column."""
    col = sbd.drop_nulls(col)
    if sbd.is_float(col):
        # avoid any issues with pandas nullable dtypes
        # (to_numpy can yield a numpy array with object dtype in old pandas
        # version if there are inf or nan)
        col = sbd.to_float32(col)
    values = sbd.to_numpy(col)
    fig, ax = plt.subplots(dpi=150)
    n_low_outliers, n_high_outliers = _robust_hist(values, ax)
    if duration_unit is not None:
        ax.set_xlabel(f"{duration_unit.capitalize()}s")
    if sbd.is_any_date(col):
        _rotate_ticklabels(ax)
    _adjust_fig_size(fig, ax, 8.0, 4.0)
    return fig, n_low_outliers, n_high_outliers


def line(x_col, y_col):
    """Line plot for a numeric column.

    ``x_col`` provides the x-axis values, ie the sorting column (corresponding
    to the report's ``order_by`` parameter). ``y_col`` is the column to plot as
    a function of x.
    """
    x = sbd.to_numpy(x_col)
    y = sbd.to_numpy(y_col)
    fig, ax = plt.subplots(dpi=150)
    _despine(ax)
    ax.plot(x, y)
    ax.set_xlabel(_utils.elide_string(x_col.name))
    if sbd.is_any_date(x_col):
        _rotate_ticklabels(ax)
    _adjust_fig_size(fig, ax, 8.0, 4.0)
    return fig


def value_counts(value_counts, n_rows, title="", color=_ORANGE):
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
    values = [_utils.elide_string(v) for v, _ in value_counts][::-1]
    counts = [c for _, c in value_counts][::-1]
    fig, ax = plt.subplots(dpi=150)
    _despine(ax)
    rects = ax.barh(list(map(str, range(len(values)))), counts, color=color)
    percent = [_utils.format_percent(c / n_rows) for c in counts]
    large_percent = [
        f"{p: >6}" if c > counts[-1] / 2 else "" for (p, c) in zip(percent, counts)
    ]
    small_percent = [
        p if c <= counts[-1] / 2 else "" for (p, c) in zip(percent, counts)
    ]

    # those are written on top of the orange bars so we write them in black
    ax.bar_label(rects, large_percent, padding=-30, color="black", fontsize=14)
    # those are written on top of the background so we write them in foreground color
    ax.bar_label(
        rects, small_percent, padding=5, color=_TEXT_COLOR_PLACEHOLDER, fontsize=14
    )

    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(list(map(str, values)))
    if title is not None:
        ax.set_title(title, fontsize=16)

    _adjust_fig_size(fig, ax, 8.0, 0.4 * len(values))
    return fig


def scatter(x_col, y_col, c_col):
    fig, ax = plt.subplots()

    sns.scatterplot(x=x_col, y=y_col, hue=c_col, ax=ax)
    if ax.legend_ is not None:
        sns.move_legend(ax, (1.05, 0.0))
    sns.move_legend(ax, (1.05, 0.0))

    return fig


def box(x_col, y_col, c_col):
    fig, ax = plt.subplots()

    sns.stripplot(
        x=x_col,
        y=y_col,
        hue=c_col,
        dodge=False,
        size=8,
        alpha=0.5,
        zorder=0,
        palette="viridis",
        ax=ax,
    )
    sns.boxplot(
        x=x_col,
        y=y_col,
        fliersize=0,
        width=0.5,
        whis=(0, 100),  # spellchecker:disable-line
        linecolor="#000000",
        linewidth=1.0,
        color="white",
        boxprops=dict(alpha=0.1),
        zorder=1,
        ax=ax,
    )
    if ax.legend_ is not None:
        sns.move_legend(ax, (1.05, 0.0))

    return fig


def heatmap(df, title=None, **kwargs):
    fig, ax = plt.subplots()

    df = df.infer_objects(copy=False).fillna(np.nan)
    df.index = [_utils.elide_string(s) for s in df.index]
    df.columns = [_utils.elide_string(s) for s in df.columns]
    _ = sns.heatmap(df, ax=ax, **kwargs)
    if title is not None:
        ax.set_title(title)

    return fig


def plot_distribution_1d(df, x_col):
    col = df[x_col]

    duration_unit = None
    if sbd.is_duration(col):
        col, duration_unit = _utils.duration_to_numeric(col)

    if sbd.is_numeric(col) or sbd.is_any_date(col):
        fig, _, _ = histogram(col, duration_unit)
    else:
        _, counter = _utils.top_k_value_counts(col, k=10)
        fig = value_counts(counter, n_rows=sbd.shape(col)[0], title=x_col)
    return fig


def _truncate_top_k(col, k=10):
    if col is None or sbd.is_numeric(col):
        return col

    # Use only the top k most frequent items of the color column
    # if it's categorical.
    _, counter = _utils.top_k_value_counts(col, k=k)
    values, _ = zip(*counter)
    other = sbd.make_column_like(col, ["other"] * sbd.shape(col)[0], name="c")
    values = (*values, np.nan)
    col = sbd.where(col, is_in(col, values), other)
    col = sbd.make_column_like(
        col,
        [_utils.elide_string(s, max_len=20) for s in sbd.to_list(col)],
        name=sbd.name(col),
    )

    return col


def _aggregate_pairwise(x_col, y_col, c_col):
    """Create a symmetric matrix by a pairwise aggregation of its columns.

    - If the color column c_col is provided, the values of the symmetric matrix are
      the mean of c_col for a given pair of (x_col, y_col) entries
    - Otherwise, the values of the symmetric matrix are the frequency of each pair
      (x_col, y_col).
    """
    # We use Pandas for pivot and groupby operations to simplify the logic.
    cols = [sbd.to_pandas(x_col), sbd.to_pandas(y_col)]
    names = [col.name for col in cols]
    kwargs = {}
    if c_col is None:
        key = "_skore_count"  # an arbitrary column name that disappear after pivoting.
        df = (
            pd.DataFrame(cols)
            .T.assign(**{key: 1})
            .groupby(names)[key]
            .sum()
            .reset_index()
            .pivot(
                columns=names[0],
                index=names[1],
                values=key,
            )
        )
        kwargs["cbar_kws"] = {"label": "total"}
    else:
        c_col = sbd.to_pandas(c_col)
        cols.append(c_col)
        key = c_col.name
        df = (
            pd.DataFrame(cols)
            .T.groupby(names)[key]
            .mean()
            .reset_index()
            .pivot(columns=names[0], index=names[1], values=key)
        )
        kwargs["cbar_kws"] = {"label": key}
    return {"df": df} | kwargs


def plot_distribution_2d(df, x_col, y_col=None, c_col=None):
    x_col = df[x_col]
    if y_col is None:
        y_col = c_col = df[c_col]
    elif c_col is None:
        y_col = df[y_col]
    else:
        y_col, c_col = df[y_col], df[c_col]

    is_x_num, is_y_num = sbd.is_numeric(x_col), sbd.is_numeric(y_col)
    if is_x_num and is_y_num:
        return scatter(x_col, y_col, _truncate_top_k(c_col))
    elif is_x_num:
        # We use a horizontal box plot to limit xlabels overlap.
        return box(x_col, _truncate_top_k(y_col), _truncate_top_k(c_col))
    elif is_y_num:
        c_col = _truncate_top_k(c_col)
        return box(y_col, _truncate_top_k(x_col), c_col)
    else:
        if not sbd.is_numeric(c_col):
            raise ValueError(
                "If 'x_col' and 'y_col' are categories, 'c_col' must be continuous."
            )
        return heatmap(**_aggregate_pairwise(x_col, y_col, c_col))


def plot_distribution(df, x_col, y_col=None, c_col=None):
    if y_col is None and c_col is None:
        # XXX: should we allow 1d plot with a hue value (c_col)?
        return plot_distribution_1d(df, x_col)
    else:
        return plot_distribution_2d(df, x_col, y_col, c_col)


def plot_pearson(df):
    pearson_table = _column_associations._compute_pearson(df)
    pearson_table = sbd.to_pandas(pearson_table).pivot(
        index="left_column_name",
        columns="right_column_name",
        values="pearson_corr",
    )
    return heatmap(pearson_table, title="Pearson Corr")


def plot_cramer(df):
    cramer_v_table = pd.DataFrame(
        _column_associations._cramer_v_matrix(df),
        columns=df.columns,
        index=df.columns,
    )
    return heatmap(cramer_v_table, title="Cramer's V")
