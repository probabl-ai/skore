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
        *,
        display_diagnostics: list[DiagnosticResult] | None = None,
    ) -> None:
        super().__init__(messages)
        self._diagnostics = tuple(diagnostics)
        self._display_diagnostics = tuple(display_diagnostics or ())

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
            if len(self._display_diagnostics) == len(self):
                items_html = "".join(
                    f"<li>{format_diagnostic_message_html(diagnostic)}</li>"
                    for diagnostic in self._display_diagnostics
                )
            else:
                items_html = "".join(f"<li>{escape(message)}</li>" for message in self)
            items_html = f'<ul style="margin:8px 0 0 18px;padding:0;">{items_html}</ul>'
        return (
            '<div style="margin:8px 0;padding:10px;border:1px solid #f97316;'
            "border-radius:4px;display:inline-block;"
            'font-family:monospace;font-size:13px;line-height:1.5;">'
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
            if skore_version < parse_version("0.15")
            else f"{skore_version.major}.{skore_version.minor}"
        )
    except PackageNotFoundError:
        url_version = "dev"
    return f"https://docs.skore.probabl.ai/{url_version}/user_guide/diagnostics.html#{docs_anchor}"


def _diagnostic_status(diagnostic: DiagnosticResult) -> str:
    if diagnostic.is_issue:
        return "issue detected"
    if not diagnostic.evaluated:
        return "not evaluated"
    return "no issue detected"


def format_diagnostic_message(diagnostic: DiagnosticResult) -> str:
    return (
        f"[{diagnostic.code}] {diagnostic.title}: {_diagnostic_status(diagnostic)}. "
        f"{diagnostic.explanation} "
        "Read our documentation for more details: "
        f"{get_diagnostics_documentation_url(docs_anchor=diagnostic.docs_anchor)}. "
        f"Mute with `ignore=['{diagnostic.code}']`."
    )


def format_diagnostic_message_html(diagnostic: DiagnosticResult) -> str:
    code = escape(diagnostic.code)
    title = escape(diagnostic.title)
    explanation = escape(diagnostic.explanation)
    docs_url = escape(
        get_diagnostics_documentation_url(docs_anchor=diagnostic.docs_anchor),
        quote=True,
    )
    return (
        f"[{code}] {title}: {_diagnostic_status(diagnostic)}. "
        f"{explanation} "
        f'Read <a href="{docs_url}" target="_blank" rel="noopener noreferrer">'
        "our documentation</a> for more details. "
        f"Mute with <code>ignore=['{code}']</code>."
    )


class ComparisonDiagnosticResults(DiagnosticResults):
    """Diagnostic results grouped by estimator for comparison reports."""

    def __init__(
        self,
        messages: list[str],
        diagnostics: list[DiagnosticResult],
        *,
        grouped: dict[str, tuple[list[str], list[DiagnosticResult]]],
    ) -> None:
        super().__init__(messages, diagnostics)
        self._grouped = grouped

    def _to_plain_text(self) -> str:
        issue_count, evaluated_count, total_count = self._summary()
        lines: list[str] = [
            f"Diagnostics: {issue_count} issue(s), "
            f"{evaluated_count}/{total_count} evaluated."
        ]
        for name, (msgs, _) in self._grouped.items():
            lines.append(f"[{name}]")
            lines.extend(f"- {m}" for m in msgs)
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        issue_count, evaluated_count, total_count = self._summary()
        header = (
            f"Diagnostics: {issue_count} issue(s), "
            f"{evaluated_count}/{total_count} evaluated."
        )
        groups_html = ""
        for name, (msgs, display_diags) in self._grouped.items():
            groups_html += (
                f'<div style="font-weight:600;margin-top:8px;">{escape(name)}</div>'
            )
            if display_diags:
                items = "".join(
                    f"<li>{format_diagnostic_message_html(d)}</li>"
                    for d in display_diags
                )
            else:
                items = "".join(f"<li>{escape(m)}</li>" for m in msgs)
            groups_html += f'<ul style="margin:4px 0 0 18px;padding:0;">{items}</ul>'
        return (
            '<div style="margin:8px 0;padding:10px;border:1px solid #f97316;'
            "border-radius:4px;display:inline-block;"
            'font-family:monospace;font-size:13px;line-height:1.5;">'
            f'<div style="font-weight:700;">{escape(header)}</div>'
            f"{groups_html}"
            "</div>"
        )

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        issue_count, evaluated_count, total_count = self._summary()
        title = (
            "Diagnostics "
            f"({issue_count} issue(s), {evaluated_count}/{total_count} evaluated)"
        )
        lines: list[str] = []
        for name, (msgs, _) in self._grouped.items():
            lines.append(f"[bold]{name}[/bold]")
            lines.extend(f"  - {m}" for m in msgs)
        yield Panel(
            "\n".join(lines) or "No diagnostics available.",
            title=title,
            border_style="orange1",
            expand=False,
        )
