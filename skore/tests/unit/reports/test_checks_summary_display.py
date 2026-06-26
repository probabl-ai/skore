from skore._sklearn._checks.base import ChecksSummaryDisplay


def display(
    check_results,
    *,
    not_applicable_codes=frozenset(),
    skipped_checks=None,
    ignored_checks=None,
    fast_mode=False,
):
    return ChecksSummaryDisplay(
        check_results=check_results,
        not_applicable_codes=set(not_applicable_codes),
        skipped_checks={} if skipped_checks is None else skipped_checks,
        ignored_checks={} if ignored_checks is None else ignored_checks,
        fast_mode=fast_mode,
    )


def display_html(*args, **kwargs):
    return display(*args, **kwargs)._repr_html_()


_MOCK_ISSUE = {
    "title": "Mock issue",
    "docs_url": "skd001-mock",
    "severity": "issue",
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
                "severity": "issue",
                "explanation": {"Ridge": "Reason A.", "Lasso": "Reason B."},
            }
        },
        not_applicable_codes={"SKDNA"},
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


def test_repr_html_skipped_and_ignored_blocks():
    """HTML repr shows skipped and ignored checks in separate blocks."""
    html = display_html(
        {},
        skipped_checks={
            "SKDSLOW": {
                "title": "Slow check",
                "docs_url": "skdslow",
                "explanation": "Skipped in fast mode (not cached).",
                "severity": "issue",
            }
        },
        ignored_checks={
            "SKDIGN": {
                "title": "Ignored check",
                "docs_url": "skdign",
                "explanation": "Muted via ignore or ignore_checks.",
                "severity": "issue",
            }
        },
        fast_mode=True,
    )
    assert "Skipped &amp; Ignored (2)" in html or "Skipped & Ignored (2)" in html
    assert "report-checks-summary-block-title" in html
    assert "<strong>Skipped</strong>" in html
    assert "<strong>Ignored</strong>" in html
    assert ">SKDSLOW</a>" in html
    assert "Skipped in fast mode (not cached)." in html
    assert ">SKDIGN</a>" in html
    assert "Muted via ignore or ignore_checks." in html


def test_frame_skipped_and_ignored_sections():
    """frame(section=...) exposes skipped and ignored rows."""
    summary = display(
        {},
        skipped_checks={
            "SKDSLOW": {
                "title": "Slow check",
                "docs_url": "skdslow",
                "explanation": "Skipped in fast mode (not cached).",
                "severity": "issue",
            }
        },
        ignored_checks={
            "SKDIGN": {
                "title": "Ignored check",
                "docs_url": "skdign",
                "explanation": "Muted via ignore or ignore_checks.",
                "severity": "issue",
            }
        },
    )
    assert set(summary.frame(section="skipped")["code"]) == {"SKDSLOW"}
    assert set(summary.frame(section="ignored")["code"]) == {"SKDIGN"}
