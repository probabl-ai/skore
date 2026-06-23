from __future__ import annotations

from abc import abstractmethod
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable
from uuid import uuid4

import pandas as pd

from skore._externals._sklearn_compat import parse_version
from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn.types import ReportType
from skore._utils.repr.html_repr import render_template

if TYPE_CHECKING:
    from skore._sklearn._base import _BaseReport


CheckCode = str


_TAB_SPECS: list[
    tuple[str, Literal["issue", "tip", "passed", "not_applicable"], str, str]
] = [
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
    (
        "Not Applicable",
        "not_applicable",
        "No checks were skipped as not applicable.",
        (
            "Checks that could not run because required data, task type, "
            "or model capabilities were missing."
        ),
    ),
]


def _check_section(
    code: CheckCode,
    check_result: dict,
    not_applicable_codes: set[CheckCode],
) -> Literal["issue", "tip", "passed", "not_applicable"]:
    if code in not_applicable_codes:
        return "not_applicable"
    if check_result["explanation"] is not None:
        return check_result["severity"]
    return "passed"


class ChecksSummaryDisplay(DisplayMixin):
    """Display for the checks summary.

    An instance of this class will be created by
    :meth:`~skore.EstimatorReport.checks.summarize`. This class should not be
    instantiated directly.

    The display object has an HTML representation organized in four tabs
    (``Issues``, ``Tips``, ``Passed``, ``Not Applicable``). The full list of
    check results is accessible via the :meth:`~ChecksSummaryDisplay.frame`
    method.

    Parameters
    ----------
    check_results : dict of str to dict
        Results of applicable checks keyed by check code
        (e.g. ``"SKD001"``). Each value is a dict with keys ``"title"``,
        ``"explanation"``, ``"severity"`` (``"issue"`` or ``"tip"``), and
        optionally ``"docs_url"``.

    not_applicable_codes : set of str
        Check codes that raised :class:`~skore.CheckNotApplicable`.

    n_ignored_codes : int
        Number of the checks that were muted via ``ignore=`` or the global
        ``ignore_checks`` configuration.
    """

    def __init__(
        self,
        check_results: dict[CheckCode, dict],
        not_applicable_codes: set[CheckCode],
        n_ignored_codes: int,
        fast_mode: bool,
    ) -> None:
        self._check_results = pd.DataFrame(
            [
                {
                    "code": code,
                    "title": check_result["title"],
                    "section": _check_section(code, check_result, not_applicable_codes),
                    "explanation": check_result["explanation"],
                    "documentation_url": _get_issue_documentation_url(check_result),
                }
                for code, check_result in check_results.items()
            ],
            columns=["code", "title", "section", "explanation", "documentation_url"],
        )
        self._n_ignored_codes = n_ignored_codes
        self._fast_mode = fast_mode

    @property
    def _header(self) -> str:
        return (
            f"Checks summary{' (fast mode)' if self._fast_mode else ''}: "
            f"{len(self.frame(section='issue'))} issue(s), "
            f"{len(self.frame(section='tip'))} tip(s), "
            f"{len(self.frame(section='passed'))} passed, "
            f"{len(self.frame(section='not_applicable'))} not applicable, "
            f"{self._n_ignored_codes} ignored."
        )

    def frame(
        self,
        section: Literal["issue", "tip", "passed", "not_applicable", "all"] = "all",
    ) -> pd.DataFrame:
        """Return check results as a DataFrame.

        Parameters
        ----------
        section : {"issue", "tip", "passed", "not_applicable", "all"}, default="all"
            Which results to include. ``"issue"`` / ``"tip"`` return only
            the matching findings; ``"passed"`` returns the checks that ran
            without reporting anything; ``"not_applicable"`` returns checks
            that could not run; ``"all"`` returns every check result.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with one row per check and columns ``"code"``,
            ``"title"``, ``"section"``, ``"explanation"``, and
            ``"documentation_url"``. The ``"explanation"`` column is ``None``
            for checks that passed without reporting anything.
        """
        match section:
            case "issue" | "tip" | "passed" | "not_applicable":
                return self._check_results.query("section == @section")
            case "all":
                return self._check_results.copy()
            case _:
                raise ValueError(f"Invalid section: {section}")

    def _repr_html_(self) -> str:
        tabs = []
        for label, section, empty_message, help_text in _TAB_SPECS:
            df = self.frame(section=section)
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
            "display/checks_summary_display.html.j2",
            {
                "container_id": f"skore-checks-summary-{uuid4().hex[:8]}",
                "header": self._header,
                "tabs": tabs,
            },
        )

    def __repr__(self) -> str:
        """Return a plain-text summary of check results."""
        if self._check_results.empty:
            return self._header + "\nAll checks were ignored."
        lines = [self._header]
        for label, section, _, _ in _TAB_SPECS:
            df = self.frame(section=section)
            if df.empty:
                continue
            lines.append(f"{label}:")
            if section == "passed":
                lines.extend(f"- [{row.code}] {row.title}" for row in df.itertuples())
            else:
                for row in df.itertuples():
                    msg = f"- [{row.code}] {row.title}"
                    if pd.notna(row.explanation):
                        msg += f". {row.explanation}"
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

    Attributes
    ----------
    code : str
        Unique identifier for this check, used in
        :meth:`~skore.EstimatorReport.checks.summarize` and ``ignore`` lists.

    title : str
        Short label shown for the finding when one is reported.

    docs_url : str or None, default=None
        Optional link or documentation anchor: a string starting with ``"http"``
        is shown as-is; otherwise it is treated as an HTML anchor fragment under
        the automated checks user guide.

    report_type : str
        Must be one of ``"cross-validation"``, ``"estimator"``,
        ``"comparison-estimator"``, or ``"comparison-cross-validation"``.

    severity : {"issue", "tip"}
        Severity of the finding. ``"issue"`` flags a modeling problem to fix;
        ``"tip"`` invites caution (e.g. on the interpretation of a result)
        without signaling a defect.

    slow : bool, default=False
        Whether the check is expensive to run (e.g. requires refits or
        permutation predictions). Slow checks are skipped when
        :meth:`~skore.EstimatorReport.checks.summarize` is called with
        ``fast_mode=True``, and are not computed by the HTML report repr
        (only cached results are surfaced).
    """

    code: CheckCode
    title: str
    report_type: ReportType
    docs_url: str | None = None
    severity: Literal["issue", "tip"]
    slow: bool = False

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
