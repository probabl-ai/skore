import numpy as np
from matplotlib import pyplot as plt
from skrub import _dataframe as sbd
from skrub._reporting import _utils

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
    fig, ax = plt.subplots(dpi=100)
    n_low_outliers, n_high_outliers = _robust_hist(values, ax)
    if duration_unit is not None:
        ax.set_xlabel(f"{duration_unit.capitalize()}s")
    if sbd.is_any_date(col):
        _rotate_ticklabels(ax)
    _adjust_fig_size(fig, ax, 4.0, 2.0)
    return fig, n_low_outliers, n_high_outliers


def line(x_col, y_col):
    """Line plot for a numeric column.

    ``x_col`` provides the x-axis values, ie the sorting column (corresponding
    to the report's ``order_by`` parameter). ``y_col`` is the column to plot as
    a function of x.
    """
    x = sbd.to_numpy(x_col)
    y = sbd.to_numpy(y_col)
    fig, ax = plt.subplots(dpi=100)
    _despine(ax)
    ax.plot(x, y)
    ax.set_xlabel(_utils.elide_string(x_col.name))
    if sbd.is_any_date(x_col):
        _rotate_ticklabels(ax)
    _adjust_fig_size(fig, ax, 4.0, 2.0)
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
