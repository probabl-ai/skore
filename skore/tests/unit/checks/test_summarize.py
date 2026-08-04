import pytest
from sklearn.linear_model import LinearRegression

from skore import evaluate
from skore._sklearn._checks.base import ChecksSummaryDisplay


def display_html(check_results, fast_mode=False):
    return ChecksSummaryDisplay(check_results, fast_mode=fast_mode)._repr_html_()


def display(check_results, fast_mode=False):
    return ChecksSummaryDisplay(check_results, fast_mode=fast_mode)


_MOCK_ISSUE = {
    "title": "Mock issue",
    "docs_url": "skd001-mock",
    "section": "issue",
}


def test_repr_html_inline_explanation():
    """HTML repr shows a string explanation on the parent line."""
    html = display_html({"SKD001": {**_MOCK_ISSUE, "explanation": "Single reason."}})
    assert ">SKD001</a>] <strong>Mock issue.</strong> Single reason." in html


def test_repr_html_groups_per_estimator_explanations():
    """HTML repr nests dict explanations under each check code."""
    html = display_html(
        {
            "SKD001": {
                **_MOCK_ISSUE,
                "explanation": {"Ridge": "Reason A.", "Lasso": "Reason B."},
            }
        }
    )
    assert ">SKD001</a>] <strong>Mock issue.</strong>" in html
    assert "<li>[Ridge] Reason A.</li>" in html
    assert "<li>[Lasso] Reason B.</li>" in html
    assert "report-checks-summary-sublist" in html


def test_repr_html_groups_not_applicable_explanations():
    """HTML repr nests per-estimator NA reasons in the not-applicable tab."""
    html = display_html(
        {
            "SKDNA": {
                "title": "Not applicable check",
                "docs_url": "skdna-mock",
                "section": "not_applicable",
                "explanation": {"Ridge": "Reason A.", "Lasso": "Reason B."},
            }
        }
    )
    assert "Not Applicable (1)" in html
    assert ">SKDNA</a>] <strong>Not applicable check.</strong>" in html
    assert "<li>[Ridge] Reason A.</li>" in html
    assert "<li>[Lasso] Reason B.</li>" in html
    assert "report-checks-summary-sublist" in html
    assert "report-checks-summary-sublist" in html


def test_repr_html_merges_estimators_with_same_explanation():
    """HTML repr groups estimators that share the same explanation."""
    html = display_html(
        {
            "SKD001": {
                **_MOCK_ISSUE,
                "explanation": {"Ridge": "Same reason.", "Lasso": "Same reason."},
            }
        }
    )
    assert ">SKD001</a>] <strong>Mock issue.</strong>" in html
    assert "<li>[Ridge, Lasso] Same reason.</li>" in html
    assert html.count("Same reason.") == 1


def test_frame_raises_for_invalid_section():
    """`frame(section=...)` rejects values outside the known set."""
    result = display({"SKD001": {**_MOCK_ISSUE, "explanation": "Single reason."}})
    with pytest.raises(ValueError, match="Invalid section"):
        result.frame(section="bogus")


def test_repr_plain_text_groups_per_estimator_explanations():
    """Plain-text repr nests dict explanations under each check code."""
    result = display(
        {
            "SKD001": {
                **_MOCK_ISSUE,
                "explanation": {"Ridge": "Reason A.", "Lasso": "Reason B."},
            }
        }
    )
    text = repr(result)
    assert "[SKD001] Mock issue." in text
    assert "Read more about this here" in text
    assert "  - [Ridge] Reason A." in text
    assert "  - [Lasso] Reason B." in text


def test_repr_plain_text_dict_explanation_without_docs_url():
    """Plain-text repr omits the doc link when docs_url is absent."""
    result = display(
        {
            "SKD001": {
                "title": "Mock issue",
                "docs_url": None,
                "section": "issue",
                "explanation": {"Ridge": "Reason A."},
            }
        }
    )
    text = repr(result)
    assert "[SKD001] Mock issue." in text
    assert "Read more about this here" not in text
    assert "  - [Ridge] Reason A." in text


def test_accessor_repr_and_html_delegate_to_fast_mode_summarize(regression_data):
    """The checks accessor's reprs mirror `summarize(fast_mode=True)`."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)
    assert repr(report.checks) == (
        f"{report.checks.summarize(fast_mode=True)!r}\n"
        "Explore available methods with .help()."
    )
    # container ids embed a random uuid, so compare structure, not exact HTML.
    assert "Fast mode is on" in report.checks._repr_html_()
    assert "Issues (" in report.checks._repr_html_()
