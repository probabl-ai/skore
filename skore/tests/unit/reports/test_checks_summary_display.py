from skore._sklearn._checks.base import ChecksSummaryDisplay


def display_html(check_results, fast_mode=False):
    return ChecksSummaryDisplay(check_results, fast_mode=fast_mode)._repr_html_()


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
