"""Helpers around the vendored numpydoc docstring scraper."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from skore._externals._docscrape import NumpyDocString, Parameter

_DEFAULT_PARAM_DESCRIPTION = "Forwarded to the underlying score function."
_FALLBACK_SUMMARY = "Registered metric."


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


def build_metric_method_docstring(
    get_method: Callable[..., Any],
    *,
    function: Callable[..., Any] | None = None,
    metric_cls: type | None = None,
    verbose_name: str | None = None,
    kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Build a numpydoc string for a dynamically exposed registry metric.

    Parameters
    ----------
    get_method : callable
        The metrics accessor ``get`` method (bound or unbound). Its signature and
        docstring define the shared parameters (e.g. ``data_source``, ``aggregate``)
        and the Returns section.

    function : callable or None, default=None
        Underlying score function whose summary and parameter docs are reused.

    metric_cls : type or None, default=None
        Metric class used as a docstring fallback when ``function`` has no summary.

    verbose_name : str or None, default=None
        Human-readable metric name used when no docstring summary is available.

    kwargs : mapping or None, default=None
        Default keyword arguments stored on the metric; documented as overridable
        extras in the Parameters section.
    """
    summary = None
    if function is not None:
        summary = docstring_summary(getattr(function, "__doc__", None))
    if not summary and metric_cls is not None:
        summary = docstring_summary(getattr(metric_cls, "__doc__", None))
    if not summary:
        summary = verbose_name or _FALLBACK_SUMMARY

    get_doc = getattr(get_method, "__doc__", None)
    get_parsed = parse_numpy_doc(get_doc)
    get_params = parameters_by_name(get_doc)
    score_params = parameters_by_name(
        getattr(function, "__doc__", None) if function is not None else None
    )

    parameters: list[Parameter] = []
    shared_names: set[str] = set()
    for param in inspect.signature(get_method).parameters.values():
        if param.name in {"self", "name"} or param.kind is param.VAR_KEYWORD:
            continue
        shared_names.add(param.name)
        if param.name in get_params:
            parameters.append(get_params[param.name])
        else:
            parameters.append(
                Parameter(
                    param.name,
                    f"default={param.default!r}",
                    ["Shared metric accessor parameter."],
                )
            )

    for key, default in (kwargs or {}).items():
        if key in shared_names:
            continue
        if key in score_params:
            description = param_description_text(score_params[key])
        else:
            description = _DEFAULT_PARAM_DESCRIPTION
        parameters.append(Parameter(key, f"default={default!r}", [description]))

    parameters.append(
        Parameter(
            "**kwargs",
            "",
            [
                "Additional keyword arguments forwarded to the underlying "
                "score function."
            ],
        )
    )

    returns = (
        get_parsed["Returns"]
        if get_parsed is not None and get_parsed["Returns"]
        else None
    )
    return build_numpy_docstring(summary, parameters, returns=returns)
