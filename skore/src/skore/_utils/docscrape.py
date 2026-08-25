"""Helpers around the vendored numpydoc docstring scraper."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from skore._externals.docscrape import NumpyDocString, Parameter

_TRAILING_DEFAULT = re.compile(r"(?:,\s*)?(?:default\s*=.*|optional)\s*\Z")


def callable_docstring(func: Any) -> str | None:
    """Return ``func.__doc__``, but only when the callable owns it.

    Callable objects without their own docstring (e.g. :func:`functools.partial`)
    inherit ``__doc__`` from their type, which describes the wrapper rather than
    the metric. Types defined in C rebuild their ``__doc__`` on each access, so
    the inherited docstring is compared by value rather than by identity.
    """
    doc = getattr(func, "__doc__", None)
    if doc is None or doc == getattr(type(func), "__doc__", None):
        return None
    return doc


@lru_cache(maxsize=256)
def _parse_cached(doc: str) -> NumpyDocString | None:
    try:
        return NumpyDocString(doc)
    except Exception:
        return None


def parse_numpy_doc(doc: str | None) -> NumpyDocString | None:
    """Parse ``doc`` with the vendored numpydoc scraper, if possible.

    Results are cached per docstring, so the returned object must be treated as
    read-only.
    """
    if not doc or not doc.strip():
        return None
    return _parse_cached(doc)


def docstring_summary(doc: str | None) -> str | None:
    """Return the leading summary from a numpydoc string, if any."""
    parsed = parse_numpy_doc(doc)
    if parsed is None or not parsed["Summary"]:
        return None
    summary = " ".join(parsed["Summary"]).strip()
    return summary


def parameters_by_name(
    doc: str | NumpyDocString | None,
) -> dict[str, Parameter]:
    """Return Parameters from ``doc`` keyed by name (stars stripped)."""
    parsed = doc if isinstance(doc, NumpyDocString) else parse_numpy_doc(doc)
    if parsed is None:
        return {}
    return {param.name.lstrip("*"): param for param in parsed["Parameters"]}


def replace_default(type_spec: str, default: Any) -> str:
    """Rewrite a numpydoc type spec so it advertises ``default``.

    Score-function docstrings include that function's own default, which can
    differ from the default the metric is registered with in skore.
    """
    stripped = _TRAILING_DEFAULT.sub("", type_spec.strip())
    if not stripped:
        return f"default={default!r}"
    return f"{stripped}, default={default!r}"


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
