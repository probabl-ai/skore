from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, runtime_checkable
from uuid import uuid4

import pandas as pd

from skore._externals._sklearn_compat import parse_version
from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn.types import ReportType
from skore._utils.repr.html_repr import render_template

if TYPE_CHECKING:
    from skore._sklearn._base import _BaseReport


CheckCode = str
CheckSource = str
CheckSources = str
CheckExplanation = str


class GroupedExplanation(TypedDict):
    source: CheckSources
    explanation: CheckExplanation


_SKIPPED_EXPLANATION = "Skipped in fast mode (not cached)."
_IGNORED_EXPLANATION = "Muted via ignore or ignore_checks."

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

_SKIPPED_IGNORED_TAB_LABEL = "Skipped & Ignored"
_SKIPPED_IGNORED_TAB_HELP = (
    "Checks that were not run because of fast mode or were muted by the user."
)
_SKIPPED_IGNORED_TAB_EMPTY = "No checks were skipped or ignored."
_SKIPPED_IGNORED_BLOCKS: list[tuple[str, Literal["skipped", "ignored"], str]] = [
    ("Skipped", "skipped", "No checks were skipped in fast mode."),
    ("Ignored", "ignored", "No checks were muted."),
]


def _rows_from_frame(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "code": row.code,
            "title": row.title,
            "explanation": (
                row.explanation
                if pd.notna(row.explanation) and isinstance(row.explanation, str)
                else None
            ),
            "grouped_explanations": (
                _group_explanations(row.explanation)
                if isinstance(row.explanation, dict)
                else None
            ),
            "documentation_url": (
                row.documentation_url if pd.notna(row.documentation_url) else None
            ),
        }
        for row in df.itertuples()
    ]


def _group_explanations(
    explanation: dict[CheckSource, CheckExplanation],
) -> list[GroupedExplanation]:
    """Merge estimators that share the same explanation for display."""
    grouped: defaultdict[CheckExplanation, list[CheckSource]] = defaultdict(list)
    for source, text in explanation.items():
        grouped[text].append(source)
    return [
        {"source": ", ".join(sources), "explanation": text}
        for text, sources in grouped.items()
    ]


def _check_section(
    code: CheckCode,
    check_result: dict,
    not_applicable_codes: set[CheckCode],
) -> Literal["issue", "tip", "passed", "not_applicable"]:
    if code in not_applicable_codes:
        return "not_applicable"
    if pd.notna(check_result.get("explanation")):
        return check_result["severity"]
    return "passed"


class ChecksSummaryDisplay(DisplayMixin):
    """Display for the checks summary.

    An instance of this class will be created by
    :meth:`~skore.EstimatorReport.checks.summarize`. This class should not be
    instantiated directly.

    The display object has an HTML representation organized in five tabs
    (``Issues``, ``Tips``, ``Passed``, ``Not Applicable``, ``Skipped & Ignored``).
    The full list of check results is accessible via the
    :meth:`~ChecksSummaryDisplay.frame` method.

    Parameters
    ----------
    check_results : dict of str to dict
        Results of applicable checks keyed by check code
        (e.g. ``"SKD001"``). Each value is a dict with keys ``"title"``,
        ``"explanation"``, ``"severity"`` (``"issue"`` or ``"tip"``), and
        optionally ``"docs_url"``. ``"explanation"`` is a string for single-report
        summaries, or a dict mapping source names to messages for aggregated
        comparison summaries.

    not_applicable_codes : set of str
        Check codes that raised :class:`~skore.CheckNotApplicable`.

    skipped_checks : dict of str to dict
        Check metadata for checks skipped in fast mode, keyed by check code.

    ignored_checks : dict of str to dict
        Check metadata for checks muted via ``ignore=`` or ``ignore_checks``.
    """

    def __init__(
        self,
        check_results: dict[CheckCode, dict],
        not_applicable_codes: set[CheckCode],
        skipped_checks: dict[CheckCode, dict],
        ignored_checks: dict[CheckCode, dict],
        fast_mode: bool,
    ) -> None:
        rows = [
            {
                "code": code,
                "title": check_result["title"],
                "section": _check_section(code, check_result, not_applicable_codes),
                "explanation": check_result["explanation"],
                "documentation_url": _get_issue_documentation_url(check_result),
            }
            for code, check_result in check_results.items()
        ]
        rows.extend(
            {
                "code": code,
                "title": check_result["title"],
                "section": "skipped",
                "explanation": check_result.get("explanation", _SKIPPED_EXPLANATION),
                "documentation_url": _get_issue_documentation_url(check_result),
            }
            for code, check_result in skipped_checks.items()
        )
        rows.extend(
            {
                "code": code,
                "title": check_result["title"],
                "section": "ignored",
                "explanation": check_result.get("explanation", _IGNORED_EXPLANATION),
                "documentation_url": _get_issue_documentation_url(check_result),
            }
            for code, check_result in ignored_checks.items()
        )
        self._check_results = pd.DataFrame(
            rows,
            columns=["code", "title", "section", "explanation", "documentation_url"],
        )
        self._fast_mode = fast_mode

    @property
    def _header(self) -> str:
        return (
            f"Checks summary{' (fast mode)' if self._fast_mode else ''}: "
            f"{len(self.frame(section='issue'))} issue(s), "
            f"{len(self.frame(section='tip'))} tip(s), "
            f"{len(self.frame(section='passed'))} passed, "
            f"{len(self.frame(section='not_applicable'))} not applicable, "
            f"{len(self.frame(section='skipped'))} skipped, "
            f"{len(self.frame(section='ignored'))} ignored."
        )

    def frame(
        self,
        section: Literal[
            "issue",
            "tip",
            "passed",
            "not_applicable",
            "skipped",
            "ignored",
            "all",
        ] = "all",
    ) -> pd.DataFrame:
        """Return check results as a DataFrame.

        Parameters
        ----------
        section : {"issue", "tip", "passed", "not_applicable", "skipped", \
                "ignored", "all"}, default="all"
            Which results to include. ``"issue"`` / ``"tip"`` return only
            the matching findings; ``"passed"`` returns the checks that ran
            without reporting anything; ``"not_applicable"`` returns checks
            that could not run; ``"skipped"`` returns checks not run in fast
            mode; ``"ignored"`` returns muted checks; ``"all"`` returns every
            check result.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with one row per check and columns ``"code"``,
            ``"title"``, ``"section"``, ``"explanation"``, and
            ``"documentation_url"``. The ``"explanation"`` column is ``None``
            for checks that passed without reporting anything. For
            issue/tip/not-applicable rows it is a string for single-report
            results, or a dict mapping source names to messages for aggregated
            comparison results (entries with a ``None`` explanation are
            omitted from the dict).
        """
        match section:
            case "issue" | "tip" | "passed" | "not_applicable" | "skipped" | "ignored":
                return self._check_results.query("section == @section")
            case "all":
                return self._check_results.copy()
            case _:
                raise ValueError(f"Invalid section: {section}")

    def _build_tabs(self) -> list[dict[str, object]]:
        tabs: list[dict[str, object]] = []
        for label, section, empty_message, help_text in _TAB_SPECS:
            df = self.frame(section=section)
            tabs.append(
                {
                    "label": label,
                    "empty_message": empty_message,
                    "help_text": help_text,
                    "rows": _rows_from_frame(df),
                }
            )
        skipped_df = self.frame(section="skipped")
        ignored_df = self.frame(section="ignored")
        tabs.append(
            {
                "label": _SKIPPED_IGNORED_TAB_LABEL,
                "empty_message": _SKIPPED_IGNORED_TAB_EMPTY,
                "help_text": _SKIPPED_IGNORED_TAB_HELP,
                "row_count": len(skipped_df) + len(ignored_df),
                "blocks": [
                    {
                        "label": label,
                        "empty_message": empty_message,
                        "rows": _rows_from_frame(self.frame(section=section)),
                    }
                    for label, section, empty_message in _SKIPPED_IGNORED_BLOCKS
                ],
            }
        )
        return tabs

    def _html_context(self, *, show_header: bool, nested: bool) -> dict:
        return {
            "container_id": f"skore-checks-summary-{uuid4().hex[:8]}",
            "header": self._header,
            "tabs": self._build_tabs(),
            "show_header": show_header,
            "nested": nested,
            "fast_mode": self._fast_mode,
        }

    def _embedded_repr_html(self) -> str:
        """HTML for embedding in a report repr (content only, no shadow DOM)."""
        return render_template(
            "display/checks_summary_display-content.html.j2",
            self._html_context(show_header=False, nested=True),
        )

    def _repr_html_(self) -> str:
        return render_template(
            "display/checks_summary_display.html.j2",
            self._html_context(show_header=True, nested=False),
        )

    def __repr__(self) -> str:
        """Return a plain-text summary of check results."""
        lines = [self._header]
        for label, section, _, _ in _TAB_SPECS:
            df = self.frame(section=section)
            if df.empty:
                continue
            lines.append(f"{label}:")
            lines.extend(self._repr_section_lines(df, section))
        skipped_df = self.frame(section="skipped")
        ignored_df = self.frame(section="ignored")
        if not skipped_df.empty or not ignored_df.empty:
            lines.append(f"{_SKIPPED_IGNORED_TAB_LABEL}:")
            if not skipped_df.empty:
                lines.append("Skipped:")
                lines.extend(self._repr_section_lines(skipped_df, "skipped"))
            if not ignored_df.empty:
                lines.append("Ignored:")
                lines.extend(self._repr_section_lines(ignored_df, "ignored"))
        lines.append("Mute a check with .checks.summarize(ignore=['<code>']).")
        return "\n".join(lines)

    def _repr_section_lines(
        self,
        df: pd.DataFrame,
        section: Literal[
            "issue", "tip", "passed", "skipped", "ignored", "not_applicable"
        ],
    ) -> list[str]:
        if section == "passed":
            return [f"- [{row.code}] {row.title}" for row in df.itertuples()]
        lines = []
        for row in df.itertuples():
            msg = f"- [{row.code}] {row.title}"
            explanation = row.explanation
            if isinstance(explanation, dict):
                if pd.notna(row.documentation_url):
                    msg += f". Read more about this here: {row.documentation_url}."
                else:
                    msg += "."
                lines.append(msg)
                lines.extend(
                    f"  - [{entry['source']}] {entry['explanation']}"
                    for entry in _group_explanations(explanation)
                )
            else:
                if pd.notna(explanation):
                    msg += f". {explanation}"
                if pd.notna(row.documentation_url):
                    msg += f" Read more about this here: {row.documentation_url}."
                lines.append(msg)
        return lines


@runtime_checkable
class Check(Protocol):
    """Protocol for defining checks.

    Each check wraps a callable that inspects a report. If the callable returns a
    non-empty string, that text is recorded as a finding under :attr:`code` with the
    given :attr:`title` and :attr:`severity`. Checks are scoped to report types via
    :attr:`report_type` so they only run on matching reports.

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

    report_type : list of {"estimator", "cross-validation", \
            "comparison-estimator", "comparison-cross-validation"}
        Report types this check applies to.

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
    report_types: list[ReportType]
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
