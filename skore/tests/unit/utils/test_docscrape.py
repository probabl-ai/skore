"""Tests for :mod:`skore._utils.docscrape`."""

from skore._externals._docscrape import Parameter
from skore._utils.docscrape import (
    build_numpy_docstring,
    docstring_summary,
    param_description_text,
    parameters_by_name,
    parse_numpy_doc,
)

NUMPYDOC_EXAMPLE = """\
Accuracy classification score.

Longer extended summary that should not be part of the short summary.

Parameters
----------
y_true : array-like of shape (n_samples,)
    Ground truth (correct) labels.
average : {'micro', 'macro', 'weighted'} or None, default=None
    Averaging strategy.
**kwargs
    Extra keyword arguments.

Returns
-------
score : float
    The computed score.
"""


def test_parse_numpy_doc_empty():
    assert parse_numpy_doc(None) is None
    assert parse_numpy_doc("") is None
    assert parse_numpy_doc("   ") is None


def test_parse_numpy_doc_sections():
    parsed = parse_numpy_doc(NUMPYDOC_EXAMPLE)
    assert parsed is not None
    assert parsed["Summary"] == ["Accuracy classification score."]
    assert [param.name for param in parsed["Parameters"]] == [
        "y_true",
        "average",
        "**kwargs",
    ]
    assert parsed["Returns"][0].name == "score"


def test_docstring_summary_from_numpydoc():
    assert docstring_summary(NUMPYDOC_EXAMPLE) == "Accuracy classification score."


def test_docstring_summary_plain_text_fallback():
    doc = "One line summary.\n\nMore details without a Parameters section."
    assert docstring_summary(doc) == "One line summary."


def test_docstring_summary_empty():
    assert docstring_summary(None) is None
    assert docstring_summary("") is None


def test_param_description_text():
    param = Parameter("average", "str or None", ["Averaging", "strategy."])
    assert param_description_text(param) == "Averaging strategy."


def test_param_description_text_default_when_empty():
    param = Parameter("cast", "bool", [])
    assert (
        param_description_text(param) == "Forwarded to the underlying score function."
    )
    assert param_description_text(param, default="Missing.") == "Missing."


def test_parameters_by_name_strips_stars():
    by_name = parameters_by_name(NUMPYDOC_EXAMPLE)
    assert set(by_name) == {"y_true", "average", "kwargs"}
    assert by_name["average"].type.startswith("{'micro'")


def test_parameters_by_name_empty_doc():
    assert parameters_by_name(None) == {}
    assert parameters_by_name("") == {}


def test_build_numpy_docstring():
    doc = build_numpy_docstring(
        "Compute a score.",
        [
            Parameter(
                "data_source",
                '{"test", "train"}, default="test"',
                ["The data source to use."],
            ),
            Parameter("**kwargs", "", ["Forwarded to the score function."]),
        ],
        returns=[Parameter("", "float", ["The metric value."])],
    )
    assert doc.startswith("Compute a score.")
    assert "Parameters" in doc
    assert "data_source" in doc
    assert "**kwargs" in doc
    assert "Returns" in doc
    assert "float" in doc
    assert docstring_summary(doc) == "Compute a score."
