from __future__ import annotations

from dataclasses import dataclass
from html import escape
from importlib.metadata import PackageNotFoundError, version

from rich.console import Console, ConsoleOptions, RenderResult
from rich.panel import Panel

from skore._externals._sklearn_compat import parse_version


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    """A detected issue from a diagnostic check.

    Each instance represents an issue that was found by a diagnostic check
    (e.g. overfitting, underfitting).

    Attributes
    ----------
    code : str
        Unique identifier for the diagnostic, e.g. `"SKD001"`

    title : str
        Short human-readable name of the diagnostic

    docs_anchor : str
        Anchor slug used to build the URL to the documentation page

    explanation : str
        Detailed message describing what was found
    """

    code: str
    title: str
    docs_anchor: str
    explanation: str


class DiagnosticResults:
    """Collection of diagnostic results returned by :meth:`Report.diagnose`.

    This object is iterable and yields human-readable message strings for every
    detected issue. When no issues are found, iterating yields a single
    "No issues were detected" message. Access the underlying
    :class:`DiagnosticResult` objects via the :attr:`diagnostics` property.

    Parameters
    ----------
    diagnostics : list of DiagnosticResult
        Detected issues produced by the report.

    checks_ran : int
        Total number of diagnostic checks that were executed.

    n_ignored : int
        Total number of diagnostic checks that were ignored.
    """

    def __init__(
        self, diagnostics: list[DiagnosticResult], checks_ran: int, n_ignored: int
    ) -> None:
        self._diagnostics = diagnostics
        self._checks_ran = checks_ran
        if diagnostics:
            self._messages = [format_diagnostic_message(d) for d in diagnostics]
        else:
            self._messages = ["No issues were detected in your report!"]
        self.header = (
            f"Diagnostics: {len(self._diagnostics)} issue(s) detected, "
            f"{self._checks_ran} check(s) ran, {n_ignored} ignored."
        )

    @property
    def diagnostics(self) -> list[DiagnosticResult]:
        """All detected diagnostic issues."""
        return self._diagnostics

    def __iter__(self):
        return iter(self._messages)

    def __len__(self):
        return len(self._messages)

    def __getitem__(self, index):
        return self._messages[index]

    def __contains__(self, item):
        return item in self._messages

    def __eq__(self, other):
        if isinstance(other, DiagnosticResults):
            return self._messages == other._messages
        if isinstance(other, (list, tuple)):
            return list(self._messages) == list(other)
        return NotImplemented

    def __bool__(self):
        return bool(self._messages)

    def _to_plain_text(self) -> str:
        return "\n".join([self.header, *[f"- {message}" for message in self._messages]])

    def _repr_html_(self) -> str:
        if self._diagnostics:
            items_html = "".join(
                f"<li>{format_diagnostic_message_html(d)}</li>"
                for d in self._diagnostics
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
        return {"text/plain": self._to_plain_text(), "text/html": self._repr_html_()}

    def __str__(self) -> str:
        return self._to_plain_text()

    def __repr__(self) -> str:
        return self._to_plain_text()

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield Panel(
            "\n".join(f"- {message}" for message in self._messages),
            title=f"Diagnostics ({self.header})",
            border_style="orange1",
            expand=False,
        )


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


def format_diagnostic_message(diagnostic: DiagnosticResult) -> str:
    return (
        f"[{diagnostic.code}] {diagnostic.title}. {diagnostic.explanation} "
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
        f"[{code}] {title}. {explanation} "
        f'Read <a href="{docs_url}" target="_blank" rel="noopener noreferrer">'
        "our documentation</a> for more details. "
        f"Mute with <code>ignore=['{code}']</code>."
    )
