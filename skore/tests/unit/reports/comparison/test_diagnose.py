import pytest

from skore._sklearn._diagnostic import DiagnosticDisplay


def test_diagnose_collects_component_issues(report, monkeypatch):
    """Check that issues from all component reports are collected."""
    report_names = list(report.reports_)
    per_report_issues = [
        {
            f"SKD{i:03d}": {
                "title": f"Mock issue {i}",
                "docs_url": f"skd{i:03d}-mock",
                "explanation": f"Issue {i} detected.",
            }
        }
        for i, _ in enumerate(report_names, start=1)
    ]
    for sub_report, issues in zip(
        report.reports_.values(), per_report_issues, strict=True
    ):
        monkeypatch.setattr(
            sub_report,
            "_get_issues",
            lambda iss=issues: (iss, set(iss)),
        )
    for attr in ("_issues_cache", "_checked_codes"):
        if hasattr(report, attr):
            delattr(report, attr)

    results = report.diagnose()
    assert isinstance(results, DiagnosticDisplay)
    for name, issues in zip(report_names, per_report_issues, strict=True):
        for code in issues:
            assert code in results.issues
            assert f"[{name}]" in results.issues[code]["explanation"]


def test_diagnose_reuses_component_cached_results(report, monkeypatch):
    """Check that check results are cached and reused."""
    report.diagnose()

    for sub_report in report.reports_.values():
        for estimator_report in getattr(sub_report, "estimator_reports_", [sub_report]):
            for check in estimator_report._checks_registry:
                monkeypatch.setattr(
                    check,
                    "check_function",
                    lambda rpt: pytest.fail("re-ran cached check"),
                )

    report.diagnose()
