from skore._sklearn._checks.base import ChecksSummaryDisplay


def _display(
    check_results,
    *,
    not_applicable_codes=frozenset(),
    fast_mode=False,
):
    return ChecksSummaryDisplay(
        check_results=check_results,
        not_applicable_codes=set(not_applicable_codes),
        n_ignored_codes=0,
        fast_mode=fast_mode,
    )


_MOCK_ISSUE = {
    "title": "Mock issue",
    "docs_url": "skd001-mock",
    "severity": "issue",
}


def test_repr_html_inline_explanation():
    """HTML repr shows a string explanation on the parent line."""
    html = _display(
        {"SKD001": {**_MOCK_ISSUE, "explanation": "Single reason."}}
    )._repr_html_()
    assert ">SKD001</a>] <b>Mock issue.</b> Single reason." in html


def test_repr_html_groups_per_estimator_explanations():
    """HTML repr nests dict explanations under each check code."""
    html = _display(
        {
            "SKD001": {
                **_MOCK_ISSUE,
                "explanation": {"Ridge": "Reason A.", "Lasso": "Reason B."},
            }
        }
    )._repr_html_()
    assert ">SKD001</a>] <b>Mock issue.</b>" in html
    assert "<li>[Ridge] Reason A.</li>" in html
    assert "<li>[Lasso] Reason B.</li>" in html
    assert "report-checks-summary-sublist" in html


def test_repr_html_groups_not_applicable_explanations():
    """HTML repr nests per-estimator NA reasons in the not-applicable tab."""
    html = _display(
        {
            "SKDNA": {
                "title": "Not applicable check",
                "docs_url": "skdna-mock",
                "severity": "issue",
                "explanation": {"Ridge": "Reason A.", "Lasso": "Reason B."},
            }
        },
        not_applicable_codes={"SKDNA"},
    )._repr_html_()
    assert "Not Applicable (1)" in html
    assert ">SKDNA</a>] <b>Not applicable check.</b>" in html
    assert "<li>[Ridge] Reason A.</li>" in html
    assert "<li>[Lasso] Reason B.</li>" in html
    assert "report-checks-summary-sublist" in html
    assert "report-checks-summary-sublist" in html


def test_repr_html_merges_estimators_with_same_explanation():
    """HTML repr groups estimators that share the same explanation."""
    html = _display(
        {
            "SKD001": {
                **_MOCK_ISSUE,
                "explanation": {"Ridge": "Same reason.", "Lasso": "Same reason."},
            }
        }
    )._repr_html_()
    assert ">SKD001</a>] <b>Mock issue.</b>" in html
    assert "<li>[Ridge, Lasso] Same reason.</li>" in html
    assert html.count("Same reason.") == 1
