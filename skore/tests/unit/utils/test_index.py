import pandas as pd
import pytest
from skore.utils._index import flatten_multiindex, unflatten_index


@pytest.mark.parametrize(
    "input_tuples, names, expected_values",
    [
        pytest.param(
            [("a", 1), ("b", 2)], ["letter", "number"], ["a_1", "b_2"], id="basic"
        ),
        pytest.param(
            [("a", 1, "x"), ("b", 2, "y")],
            ["letter", "number", "symbol"],
            ["a_1_x", "b_2_y"],
            id="multiple_levels",
        ),
        pytest.param(
            [("a", None), (None, 2)],
            ["letter", "number"],
            ["a_nan", "nan_2.0"],
            id="none_values",
        ),
        pytest.param(
            [("a@b", "1#2"), ("c&d", "3$4")],
            ["letter", "number"],
            ["a@b_1#2", "c&d_3$4"],
            id="special_chars",
        ),
        pytest.param([], ["letter", "number"], [], id="empty"),
    ],
)
def test_flatten_multiindex(input_tuples, names, expected_values):
    """Test flatten_multiindex with various input cases."""
    mi = pd.MultiIndex.from_tuples(input_tuples, names=names)
    result = flatten_multiindex(mi)
    expected = pd.Index(expected_values)
    pd.testing.assert_index_equal(result, expected)


def test_flatten_multiindex_invalid_input():
    """Test that non-MultiIndex input raises ValueError."""
    simple_index = pd.Index(["a", "b"])
    with pytest.raises(ValueError, match="`index` must be a MultiIndex."):
        flatten_multiindex(simple_index)


@pytest.mark.parametrize(
    "input_values, names, expected_tuples",
    [
        pytest.param(
            ["a_1", "b_2"], ["letter", "number"], [("a", "1"), ("b", "2")], id="basic"
        ),
        pytest.param(
            ["a_1_x", "b_2_y"],
            ["letter", "number", "symbol"],
            [("a", "1", "x"), ("b", "2", "y")],
            id="multiple_components",
        ),
        pytest.param(
            ["a_1", "b_2"], None, [("a", "1"), ("b", "2")], id="without_names"
        ),
        pytest.param(
            ["a@b_1#2", "c&d_3$4"],
            ["letter", "number"],
            [("a@b", "1#2"), ("c&d", "3$4")],
            id="special_chars",
        ),
        pytest.param([], ["letter", "number"], [], id="empty"),
    ],
)
def test_unflatten_index(input_values, names, expected_tuples):
    """Test unflatten_index with various input cases."""
    flat_idx = pd.Index(input_values)
    result = unflatten_index(flat_idx, names=names)
    expected = pd.MultiIndex.from_tuples(expected_tuples, names=names)
    pd.testing.assert_index_equal(result, expected)


def test_unflatten_index_invalid_input():
    """Test that MultiIndex input raises ValueError."""
    mi = pd.MultiIndex.from_tuples([("a", "1"), ("b", "2")])
    with pytest.raises(ValueError, match="`index` must be a flat Index."):
        unflatten_index(mi)


@pytest.mark.parametrize(
    "input_values, names, expected_names",
    [
        pytest.param(
            ["a_1", "b_2"],
            ["letter", "number"],
            ["letter", "number"],
            id="matching_names",
        ),
        pytest.param(
            ["a_1_x", "b_2_y"],
            ["level0", "level1", "level2"],
            ["level0", "level1", "level2"],
            id="three_component_names",
        ),
    ],
)
def test_unflatten_index_mismatched_names(input_values, names, expected_names):
    """Test unflatten_index with mismatched number of names."""
    flat_idx = pd.Index(input_values)
    result = unflatten_index(flat_idx, names=names)
    assert result.names == expected_names
