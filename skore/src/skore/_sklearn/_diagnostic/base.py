from __future__ import annotations

from abc import abstractmethod
from html import escape
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable
from uuid import uuid4

import pandas as pd

from skore._externals._sklearn_compat import parse_version
from skore._sklearn.types import ReportType
from skore._utils.repr.base import DisplayHelpMixin
from skore._utils.repr.html_repr import render_template

if TYPE_CHECKING:
    from skore._sklearn._base import _BaseReport


Severity = Literal["issue", "tip"]


class DiagnosticDisplay(DisplayHelpMixin):
    """Display for the diagnostic returned by :meth:`Report.diagnose`.

    A display object with an HTML representation organized in three tabs
    (``Issues``, ``Tips``, ``Passed``). The full list of detected findings is
    accessible via the :meth:`~DiagnosticDisplay.frame` method.

    Parameters
    ----------
    findings : dict of str to dict
        Detected findings produced by the report, keyed by check code
        (e.g. ``"SKD001"``). Each value is a dict with keys ``"title"``,
        ``"explanation"``, ``"severity"`` (``"issue"`` or ``"tip"``), and
        optionally ``"docs_url"``.

    checks_metadata : dict of str to dict
        Mapping from check code to a dict with keys ``"title"``, ``"docs_url"``
        and ``"severity"`` for every check known to the report.

    applicable_codes : set of str
        Codes of the checks that were applicable and ran on the report, i.e.
        those that did not raise :class:`CheckNotApplicable`.

    ignored_codes : set of str
        Codes of the checks that were muted via ``ignore=`` or the global
        ``ignore_checks`` configuration.
    """

    def __init__(
        self,
        findings: dict[str, dict],
        checks_metadata: dict[str, dict],
        applicable_codes: set[str],
        ignored_codes: set[str],
    ) -> None:
        self._findings = findings
        self._checks_metadata = checks_metadata
        self._ignored_codes = set(ignored_codes)
        self._issues = {
            code: finding
            for code, finding in findings.items()
            if finding.get("severity", "issue") == "issue"
        }
        self._tips = {
            code: finding
            for code, finding in findings.items()
            if finding.get("severity") == "tip"
        }
        self._passed = {
            code: {
                "title": checks_metadata.get(code, {}).get("title", code),
                "docs_url": checks_metadata.get(code, {}).get("docs_url"),
                "severity": checks_metadata.get(code, {}).get("severity", "issue"),
            }
            for code in sorted(
                set(applicable_codes) - set(findings) - self._ignored_codes
            )
        }
        self.header = (
            f"Diagnostic: {len(self._issues)} issue(s), "
            f"{len(self._tips)} tip(s), {len(self._passed)} passed, "
            f"{len(self._ignored_codes)} ignored."
        )

    @property
    def issues(self) -> dict[str, dict]:
        """All detected issues, keyed by check code."""
        return self._issues

    @property
    def tips(self) -> dict[str, dict]:
        """All emitted tips, keyed by check code."""
        return self._tips

    @property
    def passed(self) -> dict[str, dict]:
        """Checks that ran, were applicable, and did not report anything.

        Ignored checks are excluded.
        """
        return self._passed

    def frame(
        self,
        severity: Literal["issue", "tip", "passed", "all"] = "all",
    ) -> pd.DataFrame:
        """Return findings as a DataFrame.

        Parameters
        ----------
        severity : {"issue", "tip", "passed", "all"}, default="all"
            which findings to include. ``"issue"`` / ``"tip"`` return only
            matching findings, ``"passed"`` returns the checks that ran
            without reporting anything, and ``"all"`` concatenates the three.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with one row per finding and columns ``"code"``,
            ``"title"``, ``"severity"``, ``"explanation"``, and
            ``"documentation_url"``. The ``"explanation"`` column is ``None``
            for checks that passed without reporting anything.
        """
        records: list[dict] = []
        if severity in ("issue", "all"):
            records.extend(self._records_from(self._issues))
        if severity in ("tip", "all"):
            records.extend(self._records_from(self._tips))
        if severity in ("passed", "all"):
            records.extend(self._records_from(self._passed))
        return pd.DataFrame(
            records,
            columns=["code", "title", "severity", "explanation", "documentation_url"],
        )

    @staticmethod
    def _records_from(findings: dict[str, dict]) -> list[dict]:
        return [
            {
                "code": code,
                "title": finding["title"],
                "severity": finding.get("severity", "issue"),
                "explanation": finding.get("explanation", None),
                "documentation_url": _get_issue_documentation_url(finding),
            }
            for code, finding in findings.items()
        ]

    def _repr_html_(self) -> str:
        buckets = [
            {
                "label": "Issues",
                "entries": [
                    _format_finding_html(code, entry)
                    for code, entry in self._issues.items()
                ],
                "empty_message": "No issues were detected in your report!",
            },
            {
                "label": "Tips",
                "entries": [
                    _format_finding_html(code, entry)
                    for code, entry in self._tips.items()
                ],
                "empty_message": "No tips were emitted for your report.",
            },
            {
                "label": "Passed",
                "intro": (
                    "The following checks ran on your report without finding any issue:"
                ),
                "entries": [
                    _format_finding_html(code, entry)
                    for code, entry in self._passed.items()
                ],
                "empty_message": "No checks ran on your report.",
            },
        ]
        first_non_empty = next(
            (i for i, bucket in enumerate(buckets) if bucket["entries"]), 0
        )
        return render_template(
            "diagnostic_display.html.j2",
            {
                "container_id": f"skore-diagnostic-{uuid4().hex[:8]}",
                "header": self.header,
                "buckets": buckets,
                "first_non_empty": first_non_empty,
            },
        )

    def _repr_mimebundle_(self, **kwargs: object) -> dict[str, str]:
        return {"text/plain": self.__repr__(), "text/html": self._repr_html_()}

    def __repr__(self) -> str:
        lines = [self.header]
        for label, entries in (
            ("Issues", self._issues),
            ("Tips", self._tips),
            ("Passed", self._passed),
        ):
            if not entries:
                continue
            lines.append(f"{label}:")
            if label == "Passed":
                lines.extend(
                    f"- [{code}] {entry['title']}" for code, entry in entries.items()
                )
            else:
                lines.extend(
                    f"- {_format_finding_message(code, entry)}"
                    for code, entry in entries.items()
                )
        if not (self._issues or self._tips or self._passed):
            lines.append("- No checks were run on your report.")
        return "\n".join(lines)


@runtime_checkable
class Check(Protocol):
    """Protocol for defining diagnostic checks.

    Each check wraps a callable that inspects a report. If the callable returns a
    non-empty string, that text is recorded as a finding under :attr:`code` with the
    given :attr:`title` and :attr:`severity`. Checks are scoped to a single report
    kind via :attr:`report_type` so they only run on matching reports.

    Parameters
    ----------
    code : str
        Unique identifier for this check , used in
        :meth:`~skore.EstimatorReport.diagnose` and `ignore` lists.

    title : str
        Short label shown for the finding when one is reported.

    report_type : str
        Must be one of `"cross-validation"`, `"estimator"`,
        `"comparison-estimator"`, or `"comparison-cross-validation"`.

    docs_url : str or None, default=None
        Optional link or documentation anchor: a string starting with `"http"`
        is shown as-is; otherwise it is treated as an HTML anchor fragment under
        the automatic diagnostic user guide.

    severity : {"issue", "tip"}, default="issue"
        Severity of the finding. ``"issue"`` flags a modeling problem to fix;
        ``"tip"`` invites caution (e.g. on the interpretation of a result)
        without signaling a defect.
    """

    code: str
    title: str
    report_type: ReportType
    docs_url: str | None
    severity: Severity = "issue"

    @abstractmethod
    def check_function(self, report: _BaseReport) -> str | None:
        """Check function to run on the report and that returns an explanation string.

        Parameters
        ----------
        report : _BaseReport
            The report to run the check on.

        Returns
        -------
        str or None
            An explanation string, or None if the check did not find any
            finding.
        """


def _get_issue_documentation_url(issue: dict) -> str | None:
    docs_url = issue.get("docs_url")
    if docs_url is None:
        return None
    if docs_url.startswith("http"):
        return docs_url

    try:
        skore_version = parse_version(version("skore"))
        url_version = (
            "dev"
            if skore_version < parse_version("0.15")
            else f"{skore_version.major}.{skore_version.minor}"
        )
    except PackageNotFoundError:
        url_version = "dev"
    return f"https://docs.skore.probabl.ai/{url_version}/user_guide/automatic_diagnostic.html#{docs_url}"


def _format_finding_message(code: str, finding: dict) -> str:
    msg = f"[{code}] {finding['title']}. {finding['explanation']}"
    docs_url = _get_issue_documentation_url(finding)
    if docs_url is not None:
        msg += f" Read more about this here: {docs_url}."
    msg += f" Mute with `ignore=['{code}']`."
    return msg


def _format_finding_html(code: str, finding: dict) -> str:
    escaped_code = escape(code)
    title = escape(finding["title"])
    severity = finding.get("severity", "issue")
    if "explanation" in finding and finding["explanation"]:
        explanation = escape(finding["explanation"])
        msg = f"[{escaped_code}] {title}. {explanation}"
    else:
        msg = f"[{escaped_code}] {title}."
    docs_url = _get_issue_documentation_url(finding)
    if docs_url is not None:
        escaped_url = escape(docs_url, quote=True)
        msg += (
            f' Read more about this <a href="{escaped_url}" target="_blank"'
            ' rel="noopener noreferrer">here</a>.'
        )
    if severity in ("issue", "tip"):
        msg += f" Mute with <code>ignore=['{escaped_code}']</code>."
    return msg
