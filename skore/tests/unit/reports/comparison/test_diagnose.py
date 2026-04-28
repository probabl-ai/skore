import pytest

from skore._sklearn._diagnostic import DiagnosticDisplay


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
            lambda ignored_codes, iss=issues: (iss, set(iss)),
        )
    for attr in ("_check_results_cache", "_applicable_codes"):
        if hasattr(report, attr):
            delattr(report, attr)

    results = report.diagnosis.summarize()
    assert isinstance(results, DiagnosticDisplay)
    issues = results.frame(severity="issue").set_index("code")
    for name, per_issues in zip(report_names, per_report_issues, strict=True):
        for code in per_issues:
            assert code in issues.index
            assert f"[{name}]" in issues.loc[code, "explanation"]


def test_reuses_component_cached_results(report, monkeypatch):
    """Check that check results are cached and reused."""
    report.diagnosis.summarize()

    for sub_report in report.reports_.values():
        for estimator_report in getattr(sub_report, "estimator_reports_", [sub_report]):
            for check in estimator_report._checks_registry:
                monkeypatch.setattr(
                    check,
                    "check_function",
                    lambda rpt: pytest.fail("re-ran cached check"),
                )

    report.diagnosis.summarize()
