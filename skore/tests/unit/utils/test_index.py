import pandas as pd
import pytest
from skore._utils._index import flatten_multi_index


@pytest.mark.parametrize(
    "input_tuples, names, expected_values",
    [
        pytest.param(
            [("A", 1), ("B", 2)], ["letter", "number"], ["A_1", "B_2"], id="basic"
        ),
        pytest.param(
            [("A", 1, "X"), ("B", 2, "Y")],
            ["letter", "number", "symbol"],
            ["A_1_X", "B_2_Y"],
            id="multiple_levels",
        ),
        pytest.param(
            [("A", None), (None, 2)],
            ["letter", "number"],
            ["A_nan", "nan_2.0"],
            id="none_values",
        ),
        pytest.param(
            [("A@B", "1#2"), ("C&D", "3$4")],
            ["letter", "number"],
            ["A@B_12", "C&D_3$4"],
            id="special_chars",
        ),
        pytest.param([], ["letter", "number"], [], id="empty"),
        pytest.param(
            [("Hello World", "A B"), ("Space Test", "X Y")],
            ["text", "more"],
            ["Hello_World_A_B", "Space_Test_X_Y"],
            id="spaces",
        ),
        pytest.param(
            [("A#B#C", "1#2#3"), ("X#Y", "5#6")],
            ["text", "numbers"],
            ["ABC_123", "XY_56"],
            id="hash_symbols",
        ),
        pytest.param(
            [("UPPER", "CASE"), ("MiXeD", "cAsE")],
            ["text", "type"],
            ["UPPER_CASE", "MiXeD_cAsE"],
            id="case_sensitivity",
        ),
    ],
)
def test_flatten_multi_index(input_tuples, names, expected_values):
    """Test flatten_multi_index with various input cases using default join_str."""
    mi = pd.MultiIndex.from_tuples(input_tuples, names=names)
    result = flatten_multi_index(mi)
    expected = pd.Index(expected_values)
    pd.testing.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "input_tuples, names, join_str, expected_values",
    [
        pytest.param(
            [("A", 1), ("B", 2)],
            ["letter", "number"],
            "-",
            ["A-1", "B-2"],
            id="dash_separator",
        ),
        pytest.param(
            [("A", 1), ("B", 2)],
            ["letter", "number"],
            " ",
            ["A 1", "B 2"],
            id="space_separator",
        ),
        pytest.param(
            [("Hello World", "A B"), ("Space Test", "X Y")],
            ["text", "more"],
            "-",
            ["Hello-World-A-B", "Space-Test-X-Y"],
            id="spaces_with_dash",
        ),
        pytest.param(
            [("A#B#C", "1#2#3"), ("X#Y", "5#6")],
            ["text", "numbers"],
            "|",
            ["ABC|123", "XY|56"],
            id="hash_symbols_with_pipe",
        ),
        pytest.param(
            [("A", ""), ("B", "2"), ("", "C")],
            ["letter", "number"],
            "_",
            ["A", "B_2", "C"],
            id="empty_values_underscore",
        ),
        pytest.param(
            [("A", ""), ("B", "2"), ("", "C")],
            ["letter", "number"],
            ".",
            ["A", "B.2", "C"],
            id="empty_values_dot",
        ),
    ],
)
def test_flatten_multi_index_with_join_str(
    input_tuples, names, join_str, expected_values
):
    """Test flatten_multi_index with different join_str values."""
    mi = pd.MultiIndex.from_tuples(input_tuples, names=names)
    result = flatten_multi_index(mi, join_str=join_str)
    expected = pd.Index(expected_values)
    pd.testing.assert_index_equal(result, expected)


def test_flatten_multi_index_non_multi_index_input():
    """Test that non-MultiIndex input is returned as-is."""
    simple_index = pd.Index(["a", "b"])
    result = flatten_multi_index(simple_index)
    pd.testing.assert_index_equal(result, simple_index)


def test_flatten_multi_index_non_multi_index_with_join_str():
    """Test that non-MultiIndex input is returned as-is regardless of join_str."""
    simple_index = pd.Index(["a", "b"])
    result = flatten_multi_index(simple_index, join_str="-")
    pd.testing.assert_index_equal(result, simple_index)
