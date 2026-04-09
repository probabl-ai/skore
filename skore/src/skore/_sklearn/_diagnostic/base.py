from __future__ import annotations

from collections.abc import Callable
from html import escape
from importlib.metadata import PackageNotFoundError, version

import pandas as pd

from skore._externals._sklearn_compat import parse_version
from skore._sklearn.types import ReportType
from skore._utils.repr.base import DisplayHelpMixin


class DiagnosticDisplay(DisplayHelpMixin):
    """Display for the diagnostic returned by :meth:`Report.diagnose`.

    A display object with an HTML representation, with the full list of
    detected issues accessible via the :meth:`~DiagnosticDisplay.frame` method.

    Parameters
    ----------
    issues : dict of str to dict
        Detected issues produced by the report, keyed by check code
        (e.g. ``"SKD001"``). Each value is a dict with keys ``"title"``,
        ``"explanation"``, and optionally ``"docs_url"``.

    checks_ran : int
        Total number of checks that were executed.

    n_ignored : int
        Total number of checks that were ignored.
    """

    def __init__(
        self, issues: dict[str, dict], checks_ran: int, n_ignored: int
    ) -> None:
        self._issues = issues
        self._checks_ran = checks_ran
        if issues:
            self._messages = [
                format_issue_message(code, d) for code, d in issues.items()
            ]
        else:
            self._messages = ["No issues were detected in your report!"]
        self.header = (
            f"Diagnostic: {len(self._issues)} issue(s) detected, "
            f"{self._checks_ran} check(s) ran, {n_ignored} ignored."
        )

    @property
    def issues(self) -> dict[str, dict]:
        """All detected issues, keyed by check code."""
        return self._issues

    def frame(self) -> pd.DataFrame:
        """Return detected issues as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with one row per detected issue and columns
            ``"code"``, ``"title"``, ``"explanation"``, and
            ``"documentation_url"``.
        """
        records = [
            {
                "code": code,
                "title": issue["title"],
                "explanation": issue["explanation"],
                "documentation_url": _resolve_docs_url(issue),
            }
            for code, issue in self._issues.items()
        ]
        return pd.DataFrame(
            records, columns=["code", "title", "explanation", "documentation_url"]
        )

    def _repr_html_(self) -> str:
        if self._issues:
            items_html = "".join(
                f"<li>{format_issue_message_html(code, issue)}</li>"
                for code, issue in self._issues.items()
            )
        else:
            items_html = f"<li>{escape(self._messages[0])}</li>"
        items_html = f'<ul style="margin:8px 0 0 18px;padding:0;">{items_html}</ul>'
        return (
            '<div style="margin:8px 0;padding:10px;border:1px solid #f97316;'
            "border-radius:4px;display:inline-block;"
            'font-family:monospace;font-size:13px;line-height:1.5;">'
            f'<div style="font-weight:700;">{escape(self.header)}</div>'
            f"{items_html}"
            "</div>"
        )

    def _repr_mimebundle_(self, **kwargs: object) -> dict[str, str]:
        return {"text/plain": self.__repr__(), "text/html": self._repr_html_()}

    def __repr__(self) -> str:
        return "\n".join([self.header, *[f"- {message}" for message in self._messages]])


class Check:
    def __init__(
        self,
        function: Callable,
        code: str,
        title: str,
        docs_url: str,
        report_type: ReportType,
    ):
        self.function = function
        self.code = code
        self.title = title
        self.docs_url = docs_url
        report_types = [
            "cross-validation",
            "estimator",
            "comparison-estimator",
            "comparison-cross-validation",
        ]
        if report_type not in report_types:
            raise ValueError(
                f"report_type should be one of: {', '.join(report_types)}. "
                f"Got {report_type} instead."
            )
        self.report_type = report_type

    def run(self, report):
        explanation = self.function(report)
        if explanation:
            return {
                self.code: {
                    "title": self.title,
                    "docs_url": self.docs_url,
                    "explanation": explanation,
                }
            }
        return {}


def _resolve_docs_url(issue: dict) -> str | None:
    """Resolve the documentation URL for an issue.

    Returns ``None`` if the issue has no ``"docs_url"`` key, the raw URL if it
    starts with ``"https"``, or a full documentation URL built from the anchor
    otherwise.
    """
    docs_url = issue.get("docs_url")
    if docs_url is None:
        return None
    if docs_url.startswith("https"):
        return docs_url
    return get_issue_documentation_url(docs_anchor=docs_url)


def get_issue_documentation_url(*, docs_anchor: str) -> str:
    try:
        skore_version = parse_version(version("skore"))
        url_version = (
            "dev"
            if skore_version < parse_version("0.15")
            else f"{skore_version.major}.{skore_version.minor}"
        )
    except PackageNotFoundError:
        url_version = "dev"
    return f"https://docs.skore.probabl.ai/{url_version}/user_guide/automatic_diagnostic.html#{docs_anchor}"


def _is_external_url(issue: dict) -> bool:
    return issue.get("docs_url", "").startswith("https")


def format_issue_message(code: str, issue: dict) -> str:
    msg = f"[{code}] {issue['title']}. {issue['explanation']}"
    docs_url = _resolve_docs_url(issue)
    if docs_url is not None:
        if _is_external_url(issue):
            msg += f" Read more about this here: {docs_url}."
        else:
            msg += f" Read our documentation for more details: {docs_url}."
    msg += f" Mute with `ignore=['{code}']`."
    return msg


def format_issue_message_html(code: str, issue: dict) -> str:
    escaped_code = escape(code)
    title = escape(issue["title"])
    explanation = escape(issue["explanation"])
    msg = f"[{escaped_code}] {title}. {explanation}"
    docs_url = _resolve_docs_url(issue)
    if docs_url is not None:
        escaped_url = escape(docs_url, quote=True)
        if _is_external_url(issue):
            msg += (
                f' Read more about this <a href="{escaped_url}" target="_blank"'
                ' rel="noopener noreferrer">here</a>.'
            )
        else:
            msg += (
                f' Read <a href="{escaped_url}" target="_blank"'
                ' rel="noopener noreferrer">our documentation</a>'
                " for more details."
            )
    msg += f" Mute with <code>ignore=['{escaped_code}']</code>."
    return msg
