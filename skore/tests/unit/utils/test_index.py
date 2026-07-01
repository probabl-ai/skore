import pandas as pd
import pytest

from skore._utils._index import flatten_multi_index, squeeze_single_column


@pytest.mark.parametrize(
    "input_tuples, names, expected_values",
    [
        pytest.param(
            [("A", 1), ("B", 2)], ["letter", "number"], ["a_1", "b_2"], id="basic"
        ),
        pytest.param(
            [("A", 1, "X"), ("B", 2, "Y")],
            ["letter", "number", "symbol"],
            ["a_1_x", "b_2_y"],
            id="multiple_levels",
        ),
        pytest.param(
            [("A", None), (None, 2)],
            ["letter", "number"],
            ["a_nan", "nan_2.0"],
            id="none_values",
        ),
        pytest.param(
            [("A@B", "1#2"), ("C&D", "3$4")],
            ["letter", "number"],
            ["a@b_12", "c&d_3$4"],
            id="special_chars",
        ),
        pytest.param([], ["letter", "number"], [], id="empty"),
        pytest.param(
            [("Hello World", "A B"), ("Space Test", "X Y")],
            ["text", "more"],
            ["hello_world_a_b", "space_test_x_y"],
            id="spaces",
        ),
        pytest.param(
            [("A#B#C", "1#2#3"), ("X#Y", "5#6")],
            ["text", "numbers"],
            ["abc_123", "xy_56"],
            id="hash_symbols",
        ),
        pytest.param(
            [("UPPER", "CASE"), ("MiXeD", "cAsE")],
            ["text", "type"],
            ["upper_case", "mixed_case"],
            id="case_sensitivity",
        ),
    ],
)
def test_flatten_multi_index(input_tuples, names, expected_values):
    """Test flatten_multi_index with various input cases."""
    mi = pd.MultiIndex.from_tuples(input_tuples, names=names)
    result = flatten_multi_index(mi)
    expected = pd.Index(expected_values)
    pd.testing.assert_index_equal(result, expected)


def test_flatten_multi_index_non_multi_index():
    """Test that non-MultiIndex input is returned as-is regardless of join_str."""
    simple_index = pd.Index(["a", "b"])
    result = flatten_multi_index(simple_index, join_str="-")
    pd.testing.assert_index_equal(result, simple_index)


def test_squeeze_single_column_keeps_multi_column_frame():
    df = pd.DataFrame({"a": [1], "b": [2]}, index=["metric"])
    pd.testing.assert_frame_equal(squeeze_single_column(df), df)


def test_squeeze_single_column_flat_column():
    df = pd.DataFrame({"LogisticRegression": [0.9, 0.8]}, index=["accuracy", "recall"])
    result = squeeze_single_column(df)
    assert isinstance(result, pd.Series)
    assert result.name == "LogisticRegression"
    pd.testing.assert_series_equal(result, df["LogisticRegression"])


def test_squeeze_single_column_multiindex_column():
    df = pd.DataFrame(
        {("RandomForestClassifier", "mean"): [0.9, 0.8]},
        index=["accuracy", "recall"],
    )
    df.columns = pd.MultiIndex.from_tuples(
        [("RandomForestClassifier", "mean")], names=["Estimator", "Aggregate"]
    )
    result = squeeze_single_column(df)
    assert isinstance(result, pd.Series)
    assert result.name == "RandomForestClassifier_mean"
