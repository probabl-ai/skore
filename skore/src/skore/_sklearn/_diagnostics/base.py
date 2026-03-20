from __future__ import annotations

from dataclasses import dataclass
from html import escape
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from rich.console import Console, ConsoleOptions, RenderResult
from rich.panel import Panel

from skore._externals._sklearn_compat import parse_version

DiagnosticKind = Literal["overfitting", "underfitting", "info"]


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    code: str
    title: str
    kind: DiagnosticKind
    docs_anchor: str
    explanation: str
    is_issue: bool
    evaluated: bool = True


class DiagnosticResults(list[str]):
    def __init__(
        self,
        messages: list[str],
        diagnostics: list[DiagnosticResult],
    ) -> None:
        super().__init__(messages)
        self._diagnostics = tuple(diagnostics)

    @property
    def diagnostics(self) -> tuple[DiagnosticResult, ...]:
        return self._diagnostics

    def _summary(self) -> tuple[int, int, int]:
        issue_count = sum(diagnostic.is_issue for diagnostic in self._diagnostics)
        evaluated_count = sum(diagnostic.evaluated for diagnostic in self._diagnostics)
        return issue_count, evaluated_count, len(self._diagnostics)

    def _to_plain_text(self) -> str:
        issue_count, evaluated_count, total_count = self._summary()
        header = (
            f"Diagnostics: {issue_count} issue(s), "
            f"{evaluated_count}/{total_count} evaluated."
        )
        if not self:
            return header
        return "\n".join([header, *[f"- {message}" for message in self]])

    def _repr_html_(self) -> str:
        issue_count, evaluated_count, total_count = self._summary()
        header = (
            f"Diagnostics: {issue_count} issue(s), "
            f"{evaluated_count}/{total_count} evaluated."
        )
        if not self:
            items_html = "<div>No diagnostics available.</div>"
        else:
            items_html = "".join(f"<li>{escape(message)}</li>" for message in self)
            items_html = f'<ul style="margin:8px 0 0 18px;padding:0;">{items_html}</ul>'
        return (
            '<div style="margin:8px 0;padding:10px;border:1px solid #f97316;'
            'border-radius:4px;font-family:monospace;font-size:13px;line-height:1.5;">'
            f'<div style="font-weight:700;">{escape(header)}</div>'
            f"{items_html}"
            "</div>"
        )

    def _repr_mimebundle_(self, **kwargs: object) -> dict[str, str]:
        return {"text/plain": self._to_plain_text(), "text/html": self._repr_html_()}

    def __str__(self) -> str:
        return self._to_plain_text()

    def __repr__(self) -> str:
        return self._to_plain_text()

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        issue_count, evaluated_count, total_count = self._summary()
        title = (
            "Diagnostics "
            f"({issue_count} issue(s), {evaluated_count}/{total_count} evaluated)"
        )
        body = "No diagnostics available."
        if self:
            body = "\n".join(f"- {message}" for message in self)
        yield Panel(body, title=title, border_style="orange1", expand=False)


def get_diagnostics_documentation_url(*, docs_anchor: str) -> str:
    try:
        skore_version = parse_version(version("skore"))
        url_version = (
            "dev"
            if skore_version < parse_version("0.1")
            else f"{skore_version.major}.{skore_version.minor}"
        )
    except PackageNotFoundError:
        url_version = "dev"
    return f"https://docs.skore.probabl.ai/{url_version}/user_guide/diagnostics.html#{docs_anchor}"


def format_diagnostic_message(diagnostic: DiagnosticResult) -> str:
    status = (
        "issue detected"
        if diagnostic.is_issue
        else "not evaluated"
        if not diagnostic.evaluated
        else "no issue detected"
    )
    return (
        f"[{diagnostic.code}] {diagnostic.title}: {status}. "
        f"{diagnostic.explanation} "
        f"See {get_diagnostics_documentation_url(docs_anchor=diagnostic.docs_anchor)}. "
        f"Mute with ignore=['{diagnostic.code}']."
    )


def normalize_ignore_codes(ignore: list[str] | tuple[str, ...] | None) -> set[str]:
    if ignore is None:
        return set()
    return {code.strip().upper() for code in ignore if code.strip()}
