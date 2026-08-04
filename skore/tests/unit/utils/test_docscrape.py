"""Tests for :mod:`skore._utils.docscrape`."""

import functools

from skore._externals._docscrape import Parameter
from skore._utils.docscrape import (
    build_numpy_docstring,
    callable_docstring,
    docstring_summary,
    parameters_by_name,
    parse_numpy_doc,
    replace_default,
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


def test_docstring_summary_simple_paragraphs():
    """The scraper puts the first paragraph in Summary even without Parameters."""
    doc = "One line summary.\n\nMore details without a Parameters section."
    assert docstring_summary(doc) == "One line summary."


def test_docstring_summary_empty():
    assert docstring_summary(None) is None
    assert docstring_summary("") is None


def test_callable_docstring():
    def documented():
        """Own docstring."""

    def undocumented():
        pass

    class Callable:
        """Class docstring."""

        def __call__(self):
            pass

    assert callable_docstring(documented) == "Own docstring."
    assert callable_docstring(undocumented) is None
    assert callable_docstring(None) is None
    # Objects without their own docstring inherit it from their type.
    assert callable_docstring(functools.partial(documented)) is None
    assert callable_docstring(Callable()) is None
    # Types defined in C return a new docstring object on each access, so the
    # inherited docstring cannot be detected by identity. ``None`` is such a case
    # from Python 3.13 onwards, where ``NoneType`` gained a docstring.
    assert callable_docstring(object()) is None
    assert callable_docstring(()) is None


def test_replace_default():
    assert (
        replace_default("{'micro', 'macro'} or None, default='binary'", None)
        == "{'micro', 'macro'} or None, default=None"
    )
    assert replace_default("bool", True) == "bool, default=True"
    assert replace_default("str, optional", "raw") == "str, default='raw'"
    assert replace_default("default='uniform'", "raw") == "default='raw'"


def test_parameters_by_name_strips_stars():
    by_name = parameters_by_name(NUMPYDOC_EXAMPLE)
    assert set(by_name) == {"y_true", "average", "kwargs"}
    assert by_name["average"].type.startswith("{'micro'")


def test_parameters_by_name_from_parsed():
    parsed = parse_numpy_doc(NUMPYDOC_EXAMPLE)
    assert parameters_by_name(parsed) == parameters_by_name(NUMPYDOC_EXAMPLE)


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
