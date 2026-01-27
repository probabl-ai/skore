import matplotlib.pyplot as plt

from skore._sklearn._plot.utils import _despine_matplotlib_axis


def _decorate_matplotlib_axis(
    *,
    ax: plt.Axes,
    add_background_features: bool,
    n_features: int,
    xlabel: str,
    ylabel: str,
) -> None:
    """Decorate the matplotlib axis.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib axis to decorate.
    add_background_features : bool
        Whether to add a background color for each group of features.
    n_features : int
        The number of features to displayed.
    xlabel : str
        The label for the x-axis.
    """
    ax.axvline(x=0, color=".5", linestyle="--")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    _despine_matplotlib_axis(
        ax,
        axis_to_despine=("top", "right", "left"),
        remove_ticks=True,
        x_range=None,
        y_range=None,
    )
    if add_background_features:
        for feature_idx in range(0, n_features, 2):
            ax.axhspan(
                feature_idx - 0.5,
                feature_idx + 0.5,
                color="lightgray",
                alpha=0.1,
                zorder=0,
            )
