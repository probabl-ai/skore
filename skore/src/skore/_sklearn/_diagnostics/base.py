from __future__ import annotations

from html import escape
from importlib.metadata import PackageNotFoundError, version

import pandas as pd

from skore._externals._sklearn_compat import parse_version
from skore._utils.repr.base import DisplayHelpMixin


class DiagnosticsDisplay(DisplayHelpMixin):
    """Display for diagnostic results returned by :meth:`Report.diagnose`.

    A display object with rich and HTML representations, with the full diagnostic
    results accessible via the :meth:`~DiagnosticsDisplay.frame` method or the
    :attr:`~DiagnosticsDisplay.diagnostics` property.

    Parameters
    ----------
    diagnostics : dict of str to dict
        Detected issues produced by the report, keyed by diagnostic code
        (e.g. ``"SKD001"``). Each value is a dict with keys ``"title"``,
        ``"docs_anchor"``, and ``"explanation"``.

    checks_ran : int
        Total number of diagnostic checks that were executed.

    n_ignored : int
        Total number of diagnostic checks that were ignored.
    """

    def __init__(
        self, diagnostics: dict[str, dict], checks_ran: int, n_ignored: int
    ) -> None:
        self._diagnostics = diagnostics
        self._checks_ran = checks_ran
        if diagnostics:
            self._messages = [
                format_diagnostic_message(code, d) for code, d in diagnostics.items()
            ]
        else:
            self._messages = ["No issues were detected in your report!"]
        self.header = (
            f"Diagnostics: {len(self._diagnostics)} issue(s) detected, "
            f"{self._checks_ran} check(s) ran, {n_ignored} ignored."
        )

    @property
    def diagnostics(self) -> dict[str, dict]:
        """All detected diagnostic issues, keyed by diagnostic code."""
        return self._diagnostics

    def frame(self) -> pd.DataFrame:
        """Return diagnostic results as a DataFrame.

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
                "title": diagnostic["title"],
                "explanation": diagnostic["explanation"],
                "documentation_url": get_diagnostics_documentation_url(
                    docs_anchor=diagnostic["docs_anchor"]
                ),
            }
            for code, diagnostic in self._diagnostics.items()
        ]
        return pd.DataFrame(
            records, columns=["code", "title", "explanation", "documentation_url"]
        )

    def _repr_html_(self) -> str:
        if self._diagnostics:
            items_html = "".join(
                f"<li>{format_diagnostic_message_html(code, diagnostic)}</li>"
                for code, diagnostic in self._diagnostics.items()
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


def format_diagnostic_message(code: str, diagnostic: dict) -> str:
    return (
        f"[{code}] {diagnostic['title']}. {diagnostic['explanation']} "
        "Read our documentation for more details: "
        f"{get_diagnostics_documentation_url(docs_anchor=diagnostic['docs_anchor'])}. "
        f"Mute with `ignore=['{code}']`."
    )


def format_diagnostic_message_html(code: str, diagnostic: dict) -> str:
    escaped_code = escape(code)
    title = escape(diagnostic["title"])
    explanation = escape(diagnostic["explanation"])
    docs_url = escape(
        get_diagnostics_documentation_url(docs_anchor=diagnostic["docs_anchor"]),
        quote=True,
    )
    return (
        f"[{escaped_code}] {title}. {explanation} "
        f'Read <a href="{docs_url}" target="_blank" rel="noopener noreferrer">'
        "our documentation</a> for more details. "
        f"Mute with <code>ignore=['{escaped_code}']</code>."
    )
