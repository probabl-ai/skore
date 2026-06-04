import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from skore import evaluate
from skore._sklearn._plot.utils import (
    _adjust_fig_size,
    _downsample_thresholds_indices,
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


def test_apostrophe_in_label(binary_classification_data):
    """Non regression test for issue https://github.com/skore-ai/skore/issues/2860"""
    X, y = binary_classification_data
    labels = np.array(["A'B", "C'D"], dtype=object)
    y = labels[y]
    report = evaluate(LogisticRegression(), X, y)
    display = report.metrics.roc()
    fig = display.plot()
    legend_text = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("A'B" in text for text in legend_text)


@pytest.mark.parametrize("n_total", [0, 1, 5, 10, 1_000])
def test_downsample_thresholds_indices_none(n_total):
    """With ``max_n=None``, the helper returns ``np.arange(n_total)``."""
    indices = _downsample_thresholds_indices(n_total, None)
    np.testing.assert_array_equal(indices, np.arange(n_total))


@pytest.mark.parametrize("n_total, max_n", [(5, 5), (5, 10), (0, 2), (1, 2)])
def test_downsample_thresholds_indices_no_downsampling(n_total, max_n):
    """When ``n_total <= max_n``, all indices are kept."""
    indices = _downsample_thresholds_indices(n_total, max_n)
    np.testing.assert_array_equal(indices, np.arange(n_total))


@pytest.mark.parametrize(
    "n_total, max_n", [(10, 5), (1_000, 100), (1_234, 500), (97, 3)]
)
def test_downsample_thresholds_indices_downsampling(n_total, max_n):
    """When ``n_total > max_n``, exactly ``max_n`` indices are returned and
    endpoints (0 and ``n_total - 1``) are preserved; indices are sorted and
    strictly increasing.
    """
    indices = _downsample_thresholds_indices(n_total, max_n)
    assert indices.shape == (max_n,)
    assert indices[0] == 0
    assert indices[-1] == n_total - 1
    # The indices are sorted and strictly increasing (no duplicates) when
    # ``max_n <= n_total``.
    assert np.all(np.diff(indices) >= 1)


@pytest.mark.parametrize("max_n", [0, 1, -3])
def test_downsample_thresholds_indices_invalid(max_n):
    """`max_n_thresholds` smaller than 2 raises a clear ``ValueError``."""
    with pytest.raises(ValueError, match="must be at least 2"):
        _downsample_thresholds_indices(10, max_n)
