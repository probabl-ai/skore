"""Tests for :mod:`skore._utils.docscrape`."""

from skore._externals._docscrape import Parameter
from skore._utils.docscrape import (
    build_metric_method_docstring,
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


def _get_like(
    self,
    name: str,
    data_source: str = "test",
    aggregate: tuple[str, ...] | None = ("mean", "std"),
    **kwargs,
):
    """Get a metric value.

    Parameters
    ----------
    name : str
        Name of the metric to compute.
    data_source : {"test", "train"}, default="test"
        The data source to use.
    aggregate : {"mean", "std"} or None, default=("mean", "std")
        Aggregation across splits.

    Returns
    -------
    result : float or pandas.DataFrame
        The metric value.
    """
    return None


def test_build_metric_method_docstring_reuses_get_and_score_docs():
    def score_fn(y_true, y_pred, *, average=None):
        """Accuracy classification score.

        Parameters
        ----------
        y_true : array-like
            Ground truth.
        average : str or None, default=None
            Averaging strategy.
        """
        return 1.0

    doc = build_metric_method_docstring(
        _get_like,
        function=score_fn,
        kwargs={"average": None},
    )
    assert docstring_summary(doc) == "Accuracy classification score."
    assert "data_source" in doc
    assert "aggregate" in doc
    assert "average" in doc
    assert "Averaging strategy." in doc
    assert "**kwargs" in doc
    assert "result" in doc or "float" in doc
    # Score-function positional args must not appear as call parameters.
    params_section = doc.split("Parameters", 1)[1].split("Returns", 1)[0]
    assert "y_true" not in params_section
    assert "y_pred" not in params_section
    assert "name" not in params_section


def test_build_metric_method_docstring_fallbacks():
    class MetricWithoutDocs:
        pass

    doc = build_metric_method_docstring(
        _get_like,
        function=None,
        metric_cls=MetricWithoutDocs,
        verbose_name="Fit time (s)",
        kwargs={"cast": True},
    )
    assert docstring_summary(doc) == "Fit time (s)"
    assert "cast" in doc
    assert "Forwarded to the underlying score function." in doc
