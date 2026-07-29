"""Tests for the `.checks` accessor mechanics."""

from importlib.metadata import PackageNotFoundError
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from skrub import tabular_pipeline

from skore import Check, EstimatorReport, configuration, evaluate
from skore._sklearn._checks import base
from skore._sklearn._checks._utils import CheckNotApplicable
from skore._sklearn._checks.base import (
    ChecksSummaryDisplay,
    _get_issue_documentation_url,
)


@pytest.fixture(params=[LinearRegression(), tabular_pipeline(LinearRegression())])
def regression_report(request, regression_data):
    X, y = regression_data
    return evaluate(
        request.param,
        pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])]),
        pd.Series(y),
    )


@pytest.fixture
def cv_regression_report(regression_data):
    X, y = regression_data
    return evaluate(LinearRegression(), X, y, splitter=3)


def mock_issue(report, ignored_codes, *, fast_mode=False):
    return {
        "SKD001": {
            "title": "Mock title",
            "docs_url": "skd001-overfitting",
            "explanation": "Mock overfitting detected.",
            "section": "issue",
        }
    }


class MockCheck(Check):
    code = "TST001"
    title = "Test issue"
    report_types = ["estimator"]
    docs_url = "tst001"

    def __init__(self, has_issue: bool = True, docs_url="tst001", report_type=None):
        self.has_issue = has_issue
        self.docs_url = docs_url
        self.report_types = report_type if report_type is not None else ["estimator"]

    def check_function(self, report):
        return "Something was found." if self.has_issue else None


class TipCheck(Check):
    code = "TST002"
    title = "Tip check"
    report_types = ["estimator"]
    docs_url = "tst_tip"
    severity = "tip"

    def check_function(self, report):
        return "Be careful about this."


class SlowMockCheck(Check):
    code = "TSTSLOW"
    title = "Slow mock check"
    report_types = ["estimator"]
    docs_url = "tstslow"
    slow = True

    def __init__(self, *, fail=False):
        self.calls = 0
        self._fail = fail

    def check_function(self, report):
        if self._fail:
            raise AssertionError("slow check should not have been called")
        self.calls += 1
        return "Slow finding."


class CVCheck(Check):
    code = "CVCUSTOM"
    title = "CV-level check"
    report_types = ["cross-validation"]
    docs_url = "cvcustom"

    def check_function(self, report):
        return f"Ran on {len(report.reports_)} splits."


class EstimatorCheck(Check):
    code = "ESTCUSTOM"
    title = "Estimator-level check"
    report_types = ["estimator"]
    docs_url = "estcustom"

    def check_function(self, report):
        return "Detected on a single split."


class NotApplicableMockCheck(Check):
    code = "TSTNA"
    title = "Not applicable check"
    report_types = ["estimator"]
    docs_url = "tstna"

    def check_function(self, report):
        raise CheckNotApplicable("Mock check is not applicable.")


# add


def test_add_checks_runs_custom_check(regression_report):
    """Check that add_checks runs the custom check and includes its issue."""
    regression_report.checks.add([MockCheck(has_issue=True)])
    issues = (
        regression_report.checks.summarize().frame(section="issue").set_index("code")
    )
    assert "TST001" in issues.index
    assert issues.loc["TST001", "title"] == "Test issue"
    assert issues.loc["TST001", "documentation_url"].endswith("#tst001")
    assert issues.loc["TST001", "explanation"] == "Something was found."


def test_add_checks_cv_level(cv_regression_report):
    """Check that add_checks registers a CV-level check."""
    cv_regression_report.checks.add([CVCheck()])
    issues = (
        cv_regression_report.checks.summarize().frame(section="issue").set_index("code")
    )
    assert "CVCUSTOM" in issues.index
    assert issues.loc["CVCUSTOM", "title"] == "CV-level check"
    assert issues.loc["CVCUSTOM", "documentation_url"].endswith("#cvcustom")
    assert issues.loc["CVCUSTOM", "explanation"] == "Ran on 3 splits."


def test_add_checks_estimator_level_not_on_cv_summary(cv_regression_report):
    """Estimator-scoped custom checks do not run on the CV report summary."""
    cv_regression_report.checks.add([EstimatorCheck()])
    summary = cv_regression_report.checks.summarize().frame()
    assert "ESTCUSTOM" not in set(summary["code"])


def test_add_checks_reuses_builtin_cache(monkeypatch, regression_report):
    """Check that add_checks does not re-run already cached built-in checks."""
    regression_report.checks.summarize()

    for check in regression_report._checks_registry:
        monkeypatch.setattr(
            check, "check_function", lambda report: pytest.fail("re-ran cached check")
        )

    regression_report.checks.add([MockCheck(has_issue=True)])
    regression_report.checks.summarize()


def test_add_checks_docs_url_full(regression_report):
    """Check that a full https docs_url is preserved as-is in frame()."""
    check = MockCheck(has_issue=True, docs_url="https://example.com/my-doc")
    regression_report.checks.add([check])
    result = regression_report.checks.summarize()
    frame = result.frame()
    row = frame.query("code == 'TST001'")
    assert row["documentation_url"].iloc[0] == "https://example.com/my-doc"
    assert "Read more about this here" in repr(result)


def test_add_checks_docs_url_absent(regression_report):
    """Check that missing docs_url results in None in the frame."""
    check = MockCheck(has_issue=True, docs_url=None)
    regression_report.checks.add([check])
    result = regression_report.checks.summarize()
    frame = result.frame()
    row = frame[frame["code"] == "TST001"]
    assert row["documentation_url"].isna().all()


def test_add_checks_invalid_report_type(regression_report):
    """Check that Check raises TypeError for unsupported report_type."""
    with pytest.raises(TypeError, match="must be a non-empty list"):
        regression_report.checks.add(
            [MockCheck(has_issue=False, report_type="invalid")]
        )
    with pytest.raises(TypeError, match="unsupported values"):
        regression_report.checks.add(
            [MockCheck(has_issue=False, report_type=["invalid"])]
        )


def test_add_checks_invalid_protocol(regression_report):
    """Check that Check raises TypeError for unsupported protocol."""

    class InvalidCheck:
        code = "INVALID001"
        title = "Invalid issue"
        report_types = ["estimator"]
        docs_url = "invalid001"

    with pytest.raises(TypeError, match="is not a subclass of Check."):
        regression_report.checks.add([InvalidCheck()])


# available


def test_available_returns_code_dash_title(regression_report):
    """Check that available returns strings in 'code - title' format."""
    regression_report.checks.add([MockCheck(has_issue=True)])
    available = regression_report.checks.available()
    assert "TST001 - Test issue" in available


# remove


def test_remove_checks_excludes_results(regression_report):
    """Check that remove excludes checks from results and available checks."""
    regression_report.checks.add([MockCheck(has_issue=True)])
    assert "TST001" in set(regression_report.checks.summarize().frame()["code"])

    regression_report.checks.remove("TST001")
    assert "TST001" not in set(regression_report.checks.summarize().frame()["code"])
    assert "TST001 - Test issue" not in regression_report.checks.available()


def test_remove_clears_cache(regression_report):
    """Check that remove invalidates cached results for the removed check."""
    regression_report.checks.add([MockCheck(has_issue=True)])
    regression_report.checks.summarize()
    assert "TST001" in regression_report._check_results_cache
    assert regression_report._check_results_cache["TST001"]["section"] == "issue"

    regression_report.checks.remove("TST001")
    assert "TST001" not in regression_report._check_results_cache


def test_remove_is_case_insensitive(regression_report):
    """Check that remove matches check codes case-insensitively."""
    regression_report.checks.add([MockCheck(has_issue=True)])
    assert "TST001 - Test issue" in regression_report.checks.available()
    regression_report.checks.remove("tst001")
    assert "TST001 - Test issue" not in regression_report.checks.available()


# --------------------------------------------------------------------------- #
# summarize(): section routing (issue/tip/passed/not_applicable/ignored),
# ignore=, caching, and fast_mode/slow-check handling
# --------------------------------------------------------------------------- #


def test_no_issues(monkeypatch, regression_report):
    """Check that no issues are detected when checks pass."""
    monkeypatch.setattr(
        EstimatorReport,
        "_get_checks_results",
        lambda report, ignored_codes, *, fast_mode=False: {},
    )
    assert regression_report.checks.summarize().frame(section="issue").empty


# ignore


def test_ignore_checks(regression_report):
    """Check that checks are ignored when ignore is passed."""
    result = regression_report.checks.summarize(ignore=["SKD001"])
    assert "SKD001" in set(result.frame(section="ignored")["code"])
    assert "SKD001" not in set(result.frame(section="issue")["code"])


def test_global_ignore(regression_report):
    """Check that checks are ignored when global ignore is set."""
    assert "SKD001" not in set(
        regression_report.checks.summarize().frame(section="ignored")["code"]
    )
    with configuration(ignore_checks=["SKD001"]):
        summary = regression_report.checks.summarize()
        assert "SKD001" not in set(summary.frame(section="issue")["code"])
        assert "SKD001" in set(summary.frame(section="ignored")["code"])


# cache


def test_reuses_cached_results(monkeypatch, regression_report):
    """Check that check results are cached and reused."""
    calls = 0
    original_run = Check.check_function

    def counting_run(self, report):
        nonlocal calls
        calls += 1
        return original_run(self, report)

    monkeypatch.setattr(Check, "check_function", counting_run)
    regression_report.checks.summarize()
    calls_after_first = calls
    regression_report.checks.summarize()
    assert calls == calls_after_first


def test_reuses_cv_cached_results(monkeypatch, cv_regression_report):
    """Check that CV-level check results are cached and reused."""
    cv_regression_report.checks.summarize()

    for check in cv_regression_report._checks_registry:
        if check.code == "SKD003":
            monkeypatch.setattr(
                check,
                "check_function",
                lambda rpt: pytest.fail("re-ran cached check"),
            )

    cv_regression_report.checks.summarize()


# sections


def test_tip_goes_to_tips_not_issues(regression_report):
    """A check with section='tip' is routed to tips, not issues."""
    regression_report.checks.add([TipCheck()])
    result = regression_report.checks.summarize()
    tips = result.frame(section="tip").set_index("code")
    assert "TST002" in tips.index
    assert "TST002" not in set(result.frame(section="issue")["code"])
    assert tips.loc["TST002", "section"] == "tip"


def test_passed_contains_applicable_checks_with_no_finding(regression_report):
    """Checks that ran without reporting anything show up as passed."""
    regression_report.checks.add([MockCheck(has_issue=False)])
    result = regression_report.checks.summarize()
    assert "TST001" in set(result.frame(section="passed")["code"])
    assert "TST001" not in set(result.frame(section="issue")["code"])
    assert "TST001" not in set(result.frame(section="tip")["code"])


def test_ignored_checks_appear_in_ignored_section(regression_report):
    """Ignored codes appear under the ignored section."""
    regression_report.checks.add([MockCheck(has_issue=False)])
    result = regression_report.checks.summarize(ignore=["TST001"])
    ignored = result.frame(section="ignored").set_index("code")
    assert "TST001" in ignored.index
    assert pd.isna(ignored.loc["TST001", "explanation"])
    assert "TST001" not in set(result.frame(section="passed")["code"])
    assert "TST001" not in set(result.frame(section="issue")["code"])


def test_custom_check_not_applicable_goes_to_not_applicable_section(regression_report):
    """A check raising CheckNotApplicable appears under not applicable."""
    regression_report.checks.add([NotApplicableMockCheck()])
    result = regression_report.checks.summarize()
    na = result.frame(section="not_applicable").set_index("code")
    assert "TSTNA" in na.index
    assert na.loc["TSTNA", "explanation"] == "Mock check is not applicable."
    assert "TSTNA" not in set(result.frame(section="passed")["code"])
    assert "TSTNA" not in set(result.frame(section="issue")["code"])
    assert "TSTNA" not in set(result.frame(section="tip")["code"])


def test_frame_section_filter(regression_report):
    """`frame(section=...)` returns only rows of the requested bucket."""
    regression_report.checks.add([MockCheck(has_issue=True), TipCheck()])
    result = regression_report.checks.summarize()

    issues_frame = result.frame(section="issue")
    assert set(issues_frame["code"]) >= {"TST001"}
    assert all(issues_frame["section"] == "issue")

    tips_frame = result.frame(section="tip")
    assert set(tips_frame["code"]) >= {"TST002"}
    assert all(tips_frame["section"] == "tip")

    passed_codes = set(result.frame(section="passed")["code"])
    assert "TST001" not in passed_codes
    assert "TST002" not in passed_codes

    assert set(result.frame()["code"]) >= {"TST001", "TST002"}


# fast-mode


def test_summarize_fast_mode_skips_uncached_slow_checks(regression_report):
    """fast_mode=True skips slow checks that are not cached."""
    slow_check = SlowMockCheck()
    regression_report.checks.add([slow_check])
    result = regression_report.checks.summarize(fast_mode=True)
    assert "TSTSLOW" not in set(result.frame(section="issue")["code"])
    skipped = result.frame(section="skipped").set_index("code")
    assert "TSTSLOW" in skipped.index
    assert pd.isna(skipped.loc["TSTSLOW", "explanation"])
    assert slow_check.calls == 0


def test_summarize_fast_mode_uses_cached_slow_results(regression_report):
    """fast_mode=True surfaces slow results that were already cached."""
    slow_check = SlowMockCheck()
    regression_report.checks.add([slow_check])
    regression_report.checks.summarize()
    assert slow_check.calls == 1
    issues = (
        regression_report.checks.summarize(fast_mode=True)
        .frame(section="issue")
        .set_index("code")
    )
    assert "TSTSLOW" in issues.index
    assert slow_check.calls == 1


def test_fast_mode_skips_slow_checks_on_cv_report(cv_regression_report):
    """fast_mode=True skips slow uncached checks on a CV report."""
    summary = cv_regression_report.checks.summarize(fast_mode=True)
    slow_codes = {"SKD009", "SKD010", "SKD011", "SKD012"}
    assert slow_codes.isdisjoint(set(summary.frame(section="issue")["code"]))
    assert slow_codes.isdisjoint(set(summary.frame(section="passed")["code"]))
    assert slow_codes == set(summary.frame(section="skipped")["code"])


def test_subclass_check_without_slow_attr_treated_as_fast(regression_report):
    """Subclass of Check without `slow` inherits the protocol default."""

    class CheckNoSlowAttr(Check):
        code = "TSTFAST"
        title = "No slow attr"
        report_types = ["estimator"]
        docs_url = "tstfast"
        severity = "issue"

        def check_function(self, report):
            return "Found."

    check = CheckNoSlowAttr()
    assert check.slow is False
    regression_report.checks.add([check])
    codes = set(regression_report.checks.summarize(fast_mode=True).frame()["code"])
    assert "TSTFAST" in codes


# repr / HTML


def test_header_reports_all_counts(regression_report):
    """The header reports issue, tip, passed, NA, skipped and ignored counts."""
    regression_report.checks.add([MockCheck(has_issue=True), TipCheck()])
    result = regression_report.checks.summarize(ignore=["SKD001"])
    assert "issue(s)" in result._header
    assert "tip(s)" in result._header
    assert "passed" in result._header
    assert "not applicable" in result._header
    assert "skipped" in result._header
    assert "1 ignored" in result._header


def test_checks_summary_repr(monkeypatch, regression_report):
    """Check that the checks summary has a repr."""
    monkeypatch.setattr(EstimatorReport, "_get_checks_results", mock_issue)
    results = regression_report.checks.summarize()
    assert isinstance(results, ChecksSummaryDisplay)
    elements = [
        "Checks summary:",
        "Mock title.",
        "[SKD001]",
        "Mock overfitting detected",
    ]
    for element in elements:
        assert element in repr(results)
    bundle = results._repr_mimebundle_()
    assert "text/plain" in bundle
    assert "text/html" in bundle
    assert 'href="' in bundle["text/html"]
    assert "user_guide/automated_checks.html#" in bundle["text/html"]
    assert "Mute a check by passing" in bundle["text/html"]
    assert "report-hint-note-line" in bundle["text/html"]


def test_html_tabs(regression_report):
    """The HTML repr contains one label per bucket with its count."""
    regression_report.checks.add([MockCheck(has_issue=True), TipCheck()])
    html = regression_report.checks.summarize()._repr_html_()
    assert "Issues (" in html
    assert "Tips (" in html
    assert "Passed (" in html
    assert "Not Applicable (" in html
    assert "Skipped (" in html
    assert "Ignored (" in html


def test_checks_summary_html_note_lines(monkeypatch, regression_report):
    """HTML note shows fast-mode info and mute hint on separate lines."""
    monkeypatch.setattr(EstimatorReport, "_get_checks_results", mock_issue)
    html_fast = regression_report.checks.summarize(fast_mode=True)._repr_html_()
    assert "Fast mode is on" in html_fast
    assert "Mute a check by passing" in html_fast
    assert "report-hint-note-line" in html_fast
    assert "Checks summary (fast mode):" in html_fast

    html_full = regression_report.checks.summarize(fast_mode=False)._repr_html_()
    assert "Fast mode is on" not in html_full
    assert "Mute a check by passing" in html_full
    assert "Checks summary:" in html_full
    assert "Checks summary (fast mode):" not in html_full


def test_html_repr_does_not_compute_slow(regression_report):
    """The HTML repr never invokes slow check functions."""
    regression_report.checks.add([SlowMockCheck(fail=True)])
    fragments = regression_report._html_repr_fragments()
    assert "checks_summary" in fragments
    assert "report-checks-summary-list" in fragments["checks_summary"]
    assert "Issues (" in fragments["checks_summary"]
    assert "report-checks-nested" in fragments["checks_summary"]
    assert "Fast mode is on" in fragments["checks_summary"]
    assert "Checks summary" not in fragments["checks_summary"]


def test_html_repr_shows_cached_slow(regression_report):
    """A cached slow result is reflected in the HTML repr summary."""
    slow_check = SlowMockCheck()
    regression_report.checks.add([slow_check])
    regression_report.checks.summarize()
    fragments = regression_report._html_repr_fragments()
    checks_html = fragments["checks_summary"]
    assert ">TSTSLOW</a>" in checks_html
    assert "Issues (1)" in checks_html
    assert "Fast mode is on" in checks_html


def test_html_repr_fragments_includes_checks_detail(monkeypatch, regression_report):
    """The HTML repr fragments include per-check detail from fast-mode summary."""
    monkeypatch.setattr(EstimatorReport, "_get_checks_results", mock_issue)
    checks_html = regression_report._html_repr_fragments()["checks_summary"]
    assert "report-checks-summary-list" in checks_html
    assert "Issues (1)" in checks_html
    assert "report-checks-nested" in checks_html
    assert "Fast mode is on" in checks_html
    assert ">SKD001</a>" in checks_html
    assert "Mock title." in checks_html
    assert "Mock overfitting detected." in checks_html
    assert "user_guide/automated_checks.html#" in checks_html
    assert "Read more about this" not in checks_html


# _get_issue_documentation_url


def test_documentation_url_points_to_existing_rst():
    """Check that the URL in _get_issue_documentation_url maps to a real RST file."""
    url = urlparse(
        _get_issue_documentation_url(
            mock_issue(report=None, ignored_codes=set())["SKD001"]
        )
    )
    # url.path is e.g. "/dev/user_guide/automated_checks.html"
    # strip version prefix and convert .html -> .rst
    rst_rel_path = "/".join(url.path.split("/")[2:]).replace(".html", ".rst")
    rst_path = Path(__file__).parents[4] / "sphinx" / rst_rel_path
    assert rst_path.is_file()


def test_documentation_url_falls_back_to_dev_when_package_not_found(monkeypatch):
    """The docs URL uses the 'dev' version when skore's package metadata is missing."""

    def raise_not_found(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(base, "version", raise_not_found)
    url = _get_issue_documentation_url({"docs_url": "skd001-mock"})
    assert (
        url
        == "https://docs.skore.probabl.ai/dev/user_guide/automated_checks.html#skd001-mock"
    )
