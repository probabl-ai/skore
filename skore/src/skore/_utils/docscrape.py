"""Helpers around the vendored numpydoc docstring scraper."""

from __future__ import annotations

import inspect

from skore._externals._docscrape import NumpyDocString, Parameter

_DEFAULT_PARAM_DESCRIPTION = "Forwarded to the underlying score function."


def parse_numpy_doc(doc: str | None) -> NumpyDocString | None:
    """Parse ``doc`` with the vendored numpydoc scraper, if possible."""
    if not doc or not str(doc).strip():
        return None
    try:
        return NumpyDocString(doc)
    except Exception:
        return None


def docstring_summary(doc: str | None) -> str | None:
    """Return the leading summary paragraph of a docstring, if any."""
    parsed = parse_numpy_doc(doc)
    if parsed is not None:
        summary = " ".join(
            line.strip() for line in parsed["Summary"] if line and line.strip()
        ).strip()
        if summary:
            return summary

    if not doc:
        return None
    first_para = inspect.cleandoc(doc).split("\n\n", 1)[0].strip()
    if not first_para:
        return None
    return " ".join(line.strip() for line in first_para.splitlines() if line.strip())


def param_description_text(
    param: Parameter,
    *,
    default: str = _DEFAULT_PARAM_DESCRIPTION,
) -> str:
    """Join a ``Parameter.desc`` list into a single description string."""
    desc = " ".join(line.strip() for line in param.desc if line.strip())
    return desc or default


def parameters_by_name(doc: str | None) -> dict[str, Parameter]:
    """Return Parameters from ``doc`` keyed by name (stars stripped)."""
    parsed = parse_numpy_doc(doc)
    if parsed is None:
        return {}
    return {param.name.lstrip("*"): param for param in parsed["Parameters"]}


def build_numpy_docstring(
    summary: str,
    parameters: list[Parameter],
    returns: list[Parameter] | None = None,
) -> str:
    """Assemble a numpydoc string from summary, parameters and optional returns."""
    constructed = NumpyDocString("")
    constructed["Summary"] = [summary]
    constructed["Parameters"] = parameters
    if returns:
        constructed["Returns"] = returns
    return str(constructed).strip()
