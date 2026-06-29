import pytest

from skore._sklearn._checks.base import ChecksSummaryDisplay


def test_collects_component_issues(report, monkeypatch):
    """Check that issues from all component reports are collected."""
    report_names = list(report.reports_)
    per_report_issues = [
        {
            f"SKD{i:03d}": {
                "title": f"Mock issue {i}",
                "docs_url": f"skd{i:03d}-mock",
                "explanation": f"Issue {i} detected.",
                "severity": "issue",
            }
        }
        for i, _ in enumerate(report_names, start=1)
    ]
    for sub_report, issues in zip(
        report.reports_.values(), per_report_issues, strict=True
    ):
        monkeypatch.setattr(
            sub_report,
            "_get_results",
            lambda ignored_codes, *, fast_mode=False, iss=issues: (
                iss,
                set(iss),
                set(),
                set(),
            ),
        )
    for attr in ("_check_results_cache", "_applicable_codes", "_not_applicable_codes"):
        if hasattr(report, attr):
            delattr(report, attr)

    results = report.checks.summarize()
    assert isinstance(results, ChecksSummaryDisplay)
    issues = results.frame(section="issue").set_index("code")
    for name, per_issues in zip(report_names, per_report_issues, strict=True):
        for code in per_issues:
            assert code in issues.index
            assert name in issues.loc[code, "explanation"]


def test_reuses_component_cached_results(report, monkeypatch):
    """Check that check results are cached and reused."""
    report.checks.summarize()

    for sub_report in report.reports_.values():
        for estimator_report in getattr(sub_report, "reports_", [sub_report]):
            for check in estimator_report._checks_registry:
                monkeypatch.setattr(
                    check,
                    "check_function",
                    lambda rpt: pytest.fail("re-ran cached check"),
                )

    report.checks.summarize()


def test_fast_mode_surfaces_skipped_slow_checks(
    comparison_estimator_reports_regression,
):
    """fast_mode surfaces slow checks skipped on all estimators."""
    summary = comparison_estimator_reports_regression.checks.summarize(fast_mode=True)
    skipped = summary.frame(section="skipped").set_index("code")
    slow_codes = {"SKD009", "SKD010", "SKD011", "SKD012"}
    assert slow_codes <= set(skipped.index)
    for code in slow_codes:
        assert isinstance(skipped.loc[code, "explanation"], dict)
        assert set(skipped.loc[code, "explanation"]) == set(
            comparison_estimator_reports_regression.reports_
        )


def test_ignore_surfaces_muted_checks(comparison_estimator_reports_regression):
    """Ignored checks appear in the ignored section for comparison reports."""
    summary = comparison_estimator_reports_regression.checks.summarize(
        ignore=["SKD001"], fast_mode=True
    )
    ignored = summary.frame(section="ignored").set_index("code")
    assert "SKD001" in ignored.index
    assert ignored.loc["SKD001", "explanation"] == "Muted via ignore or ignore_checks."
    assert "SKD001" not in set(summary.frame(section="issue")["code"])
