from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

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
