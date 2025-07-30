import numpy as np
import pytest
from skore._sklearn._plot.utils import (
    _adjust_fig_size,
    _get_adjusted_fig_size,
    _rotate_ticklabels,
    _validate_style_kwargs,
)


@pytest.mark.parametrize(
    "default_kwargs, user_kwargs, expected",
    [
        (
            {"color": "blue", "linewidth": 2},
            {"linestyle": "dashed"},
            {"color": "blue", "linewidth": 2, "linestyle": "dashed"},
        ),
        (
            {"color": "blue", "linestyle": "solid"},
            {"c": "red", "ls": "dashed"},
            {"color": "red", "linestyle": "dashed"},
        ),
        (
            {"label": "xxx", "color": "k", "linestyle": "--"},
            {"ls": "-."},
            {"label": "xxx", "color": "k", "linestyle": "-."},
        ),
        ({}, {}, {}),
        (
            {},
            {
                "ls": "dashed",
                "c": "red",
                "ec": "black",
                "fc": "yellow",
                "lw": 2,
                "mec": "green",
                "mfcalt": "blue",
                "ms": 5,
            },
            {
                "linestyle": "dashed",
                "color": "red",
                "edgecolor": "black",
                "facecolor": "yellow",
                "linewidth": 2,
                "markeredgecolor": "green",
                "markerfacecoloralt": "blue",
                "markersize": 5,
            },
        ),
    ],
)
def test_validate_style_kwargs(default_kwargs, user_kwargs, expected):
    """Check the behaviour of `validate_style_kwargs` with various type of entries."""
    result = _validate_style_kwargs(default_kwargs, user_kwargs)
    assert result == expected, (
        "The validation of style keywords does not provide the expected results: "
        f"Got {result} instead of {expected}."
    )


@pytest.mark.parametrize(
    "default_kwargs, user_kwargs",
    [({}, {"ls": 2, "linestyle": 3}), ({}, {"c": "r", "color": "blue"})],
)
def test_validate_style_kwargs_error(default_kwargs, user_kwargs):
    """Check that `validate_style_kwargs` raises TypeError"""
    with pytest.raises(TypeError):
        _validate_style_kwargs(default_kwargs, user_kwargs)


@pytest.mark.parametrize(
    "rotation, horizontalalignment", [(45, "right"), (90, "center"), (180, "left")]
)
def test_rotate_ticklabels(pyplot, rotation, horizontalalignment):
    """Check that we can rotate the ticks labels on an axis."""
    _, ax = pyplot.subplots()
    ax.plot(range(10))
    _rotate_ticklabels(ax, rotation=rotation, horizontalalignment=horizontalalignment)
    assert ax.get_xticklabels()[0].get_rotation() == rotation
    assert ax.get_xticklabels()[0].get_horizontalalignment() == horizontalalignment


def test_get_adjusted_fig_size(pyplot):
    """Check the computation of the adjusted figure size."""
    fig, ax = pyplot.subplots(figsize=(2, 4))
    assert _get_adjusted_fig_size(fig, ax, "width", 2) == pytest.approx(
        2.0 * _get_adjusted_fig_size(fig, ax, "width", 1)
    )
    assert _get_adjusted_fig_size(fig, ax, "height", 2) == pytest.approx(
        2.0 * _get_adjusted_fig_size(fig, ax, "height", 1)
    )


def test_adjust_fig_size(pyplot):
    """Check the adjustment of the figure size."""
    fig, ax = pyplot.subplots(figsize=(2, 4))
    _adjust_fig_size(fig, ax, 1, 2)
    expected_width = _get_adjusted_fig_size(fig, ax, "width", 1)
    expected_height = _get_adjusted_fig_size(fig, ax, "height", 2)
    np.testing.assert_allclose(fig.get_size_inches(), (expected_width, expected_height))
