from __future__ import annotations

from abc import abstractmethod
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


CheckCode = str


_TAB_SPECS: list[tuple[str, Literal["issue", "tip", "passed"], str, str]] = [
    (
        "Issues",  # Title of the tab
        "issue",  # Severity of the checks collected in the tab
        "No issues were detected in your report.",  # Message when tab is empty
        "Modeling problems flagged by applicable checks.",  # Help text
    ),
    (
        "Tips",
        "tip",
        "No tips were emitted for your report.",
        "Advice to keep in mind when interpreting the results.",
    ),
    (
        "Passed",
        "passed",
        "No checks passed without reporting anything.",
        "Checks that ran on your report without flagging anything.",
    ),
]


class ChecksSummaryDisplay(DisplayHelpMixin):
    """Display for the checks summary.

    An instance of this class will be created by
    :meth:`~skore.EstimatorReport.checks.summarize`.This class should not be
    instantiated directly.

    The display object has an HTML representation organized in three tabs
    (``Issues``, ``Tips``, ``Passed``). The full list of check results is
    accessible via the :meth:`~ChecksSummaryDisplay.frame` method.

    Parameters
    ----------
    check_results : dict of str to dict
        Results of applicable checks keyed by check code
        (e.g. ``"SKD001"``). Each value is a dict with keys ``"title"``,
        ``"explanation"``, ``"severity"`` (``"issue"`` or ``"tip"``), and
        optionally ``"docs_url"``.

    n_ignored_codes : int
        Number of the checks that were muted via ``ignore=`` or the global
        ``ignore_checks`` configuration.
    """

    def __init__(
        self,
        check_results: dict[CheckCode, dict],
        n_ignored_codes: int,
    ) -> None:
        self._check_results = pd.DataFrame(
            [
                {
                    "code": code,
                    "title": check_result["title"],
                    "severity": check_result["severity"],
                    "explanation": check_result["explanation"],
                    "documentation_url": _get_issue_documentation_url(check_result),
                }
                for code, check_result in check_results.items()
            ],
            columns=["code", "title", "severity", "explanation", "documentation_url"],
        )
        self._n_ignored_codes = n_ignored_codes

    @property
    def _header(self) -> str:
        return (
            f"Checks summary: {len(self.frame(severity='issue'))} issue(s), "
            f"{len(self.frame(severity='tip'))} tip(s), "
            f"{len(self.frame(severity='passed'))} passed, "
            f"{self._n_ignored_codes} ignored."
        )

    def frame(
        self,
        severity: Literal["issue", "tip", "passed", "all"] = "all",
    ) -> pd.DataFrame:
        """Return check results as a DataFrame.

        Parameters
        ----------
        severity : {"issue", "tip", "passed", "all"}, default="all"
            Which results to include. ``"issue"`` / ``"tip"`` return only
            the matching findings (explanation is not ``None``); ``"passed"``
            returns the checks that ran without reporting anything; ``"all"``
            returns every applicable check result.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with one row per check and columns ``"code"``,
            ``"title"``, ``"severity"``, ``"explanation"``, and
            ``"documentation_url"``. The ``"explanation"`` column is ``None``
            for checks that passed without reporting anything.
        """
        match severity:
            case "issue" | "tip":
                return self._check_results.query(
                    "severity == @severity and explanation.notna()"
                )
            case "passed":
                return self._check_results.query("explanation.isna()")
            case "all":
                return self._check_results.copy()
            case _:
                raise ValueError(f"Invalid severity: {severity}")

    def _repr_html_(self) -> str:
        tabs = []
        for label, severity, empty_message, help_text in _TAB_SPECS:
            df = self.frame(severity=severity)
            tabs.append(
                {
                    "label": label,
                    "empty_message": empty_message,
                    "help_text": help_text,
                    "rows": [
                        {
                            "code": row.code,
                            "title": row.title,
                            "explanation": row.explanation
                            if pd.notna(row.explanation)
                            else None,
                            "documentation_url": row.documentation_url
                            if pd.notna(row.documentation_url)
                            else None,
                        }
                        for row in df.itertuples()
                    ],
                }
            )
        return render_template(
            "checks_summary_display.html.j2",
            {
                "container_id": f"skore-checks-summary-{uuid4().hex[:8]}",
                "header": self._header,
                "tabs": tabs,
            },
        )

    def _repr_mimebundle_(self, **kwargs: object) -> dict[str, str]:
        return {"text/plain": self.__repr__(), "text/html": self._repr_html_()}

    def __repr__(self) -> str:
        if self._check_results.empty:
            return self._header + "\nAll checks were either ignored or not applicable."
        lines = [self._header]
        for label, severity, _, _ in _TAB_SPECS:
            df = self.frame(severity=severity)
            if df.empty:
                continue
            lines.append(f"{label}:")
            if severity == "passed":
                lines.extend(f"- [{row.code}] {row.title}" for row in df.itertuples())
            else:
                for row in df.itertuples():
                    msg = f"- [{row.code}] {row.title}. {row.explanation}"
                    if pd.notna(row.documentation_url):
                        msg += f" Read more about this here: {row.documentation_url}."
                    lines.append(msg)
        lines.append("Mute a check with .checks.summarize(ignore=['<code>']).")
        return "\n".join(lines)


@runtime_checkable
class Check(Protocol):
    """Protocol for defining checks.

    Each check wraps a callable that inspects a report. If the callable returns a
    non-empty string, that text is recorded as a finding under :attr:`code` with the
    given :attr:`title` and :attr:`severity`. Checks are scoped to a single report
    type via :attr:`report_type` so they only run on matching reports.

    Parameters
    ----------
    code : str
        Unique identifier for this check , used in
        :meth:`~skore.EstimatorReport.checks.summarize` and `ignore` lists.

    title : str
        Short label shown for the finding when one is reported.

    report_type : str
        Must be one of `"cross-validation"`, `"estimator"`,
        `"comparison-estimator"`, or `"comparison-cross-validation"`.

    docs_url : str or None, default=None
        Optional link or documentation anchor: a string starting with `"http"`
        is shown as-is; otherwise it is treated as an HTML anchor fragment under
        the automated checks user guide.

    severity : {"issue", "tip"}
        Severity of the finding. ``"issue"`` flags a modeling problem to fix;
        ``"tip"`` invites caution (e.g. on the interpretation of a result)
        without signaling a defect.
    """

    code: CheckCode
    title: str
    report_type: ReportType
    docs_url: str | None
    severity: Literal["issue", "tip"]

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
            An explanation string, or None if the check did not find anything.
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
    return f"https://docs.skore.probabl.ai/{url_version}/user_guide/automated_checks.html#{docs_url}"
