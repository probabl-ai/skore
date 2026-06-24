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
            explanation = issues.loc[code, "explanation"]
            assert isinstance(explanation, dict)
            assert name in explanation


def test_html_nested_findings(report, monkeypatch):
    """HTML repr nests per-estimator findings under each check code."""
    report_names = list(report.reports_)
    assert len(report_names) >= 2
    name_a, name_b = report_names[:2]

    def make_get_results(issues):
        return lambda ignored_codes, *, fast_mode=False: (
            issues,
            set(issues),
            set(),
        )

    for name, sub_report in report.reports_.items():
        if name == name_a:
            issues = {
                "SKD001": {
                    "title": "Mock issue",
                    "docs_url": "skd001-mock",
                    "explanation": "Reason A.",
                    "severity": "issue",
                }
            }
        elif name == name_b:
            issues = {
                "SKD001": {
                    "title": "Mock issue",
                    "docs_url": "skd001-mock",
                    "explanation": "Reason B.",
                    "severity": "issue",
                }
            }
        else:
            issues = {}
        monkeypatch.setattr(sub_report, "_get_results", make_get_results(issues))

    for attr in ("_check_results_cache", "_applicable_codes", "_not_applicable_codes"):
        if hasattr(report, attr):
            delattr(report, attr)

    html = report.checks.summarize()._repr_html_()
    assert ">SKD001</a>" in html
    assert "Mock issue." in html
    assert f"[{name_a}]" in html
    assert f"[{name_b}]" in html
    assert "Reason A." in html
    assert "Reason B." in html
    assert "Detected in:" not in html
    assert "report-checks-summary-sublist" in html


def test_html_nested_not_applicable_findings(report, monkeypatch):
    """HTML repr nests per-estimator NA reasons under each check code."""
    report_names = list(report.reports_)
    assert len(report_names) >= 2
    name_a, name_b = report_names[:2]

    def make_get_results(results, applicable, not_applicable):
        return lambda ignored_codes, *, fast_mode=False: (
            results,
            applicable,
            not_applicable,
        )

    for name, sub_report in report.reports_.items():
        if name == name_a:
            results = {
                "SKDNA": {
                    "title": "Not applicable check",
                    "docs_url": "skdna-mock",
                    "explanation": "Reason A.",
                    "severity": "issue",
                }
            }
            monkeypatch.setattr(
                sub_report,
                "_get_results",
                make_get_results(results, set(), {"SKDNA"}),
            )
        elif name == name_b:
            results = {
                "SKDNA": {
                    "title": "Not applicable check",
                    "docs_url": "skdna-mock",
                    "explanation": "Reason B.",
                    "severity": "issue",
                }
            }
            monkeypatch.setattr(
                sub_report,
                "_get_results",
                make_get_results(results, set(), {"SKDNA"}),
            )
        else:
            monkeypatch.setattr(
                sub_report,
                "_get_results",
                make_get_results({}, set(), set()),
            )

    for attr in ("_check_results_cache", "_applicable_codes", "_not_applicable_codes"):
        if hasattr(report, attr):
            delattr(report, attr)

    results = report.checks.summarize()
    html = results._repr_html_()
    na = results.frame(section="not_applicable").set_index("code")
    assert "SKDNA" in na.index
    assert na.loc["SKDNA", "explanation"] == {name_a: "Reason A.", name_b: "Reason B."}
    assert f"[{name_a}]" in html
    assert f"[{name_b}]" in html
    assert "Reason A." in html
    assert "Reason B." in html
    assert "Not Applicable (1)" in html
    assert "report-checks-summary-sublist" in html


def test_html_contracts_shared_findings(report, monkeypatch):
    """HTML repr groups estimators that share the same explanation."""
    report_names = list(report.reports_)
    assert len(report_names) >= 2
    name_a, name_b = report_names[:2]
    shared_reason = "Same reason for both estimators."

    def make_get_results(issues):
        return lambda ignored_codes, *, fast_mode=False: (
            issues,
            set(issues),
            set(),
        )

    shared_issue = {
        "SKD001": {
            "title": "Mock issue",
            "docs_url": "skd001-mock",
            "explanation": shared_reason,
            "severity": "issue",
        }
    }
    for name, sub_report in report.reports_.items():
        if name in {name_a, name_b}:
            monkeypatch.setattr(
                sub_report, "_get_results", make_get_results(shared_issue)
            )
        else:
            monkeypatch.setattr(sub_report, "_get_results", make_get_results({}))

    for attr in ("_check_results_cache", "_applicable_codes", "_not_applicable_codes"):
        if hasattr(report, attr):
            delattr(report, attr)

    html = report.checks.summarize()._repr_html_()
    assert f"<li>[{name_a}, {name_b}] {shared_reason}</li>" in html
    assert html.count(shared_reason) == 1


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
