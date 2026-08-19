from __future__ import annotations

import inspect
from abc import abstractmethod
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import version
from keyword import iskeyword
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeVar, cast

import pandas as pd

from skore._externals._docscrape import Parameter
from skore.checks._utils import CheckNotApplicable
from skore.checks.base import Check, CheckCode, CheckResult, CheckSection
from skore.checks.model_checks import _BUILTIN_CHECKS
from skore.metrics import Metric
from skore.project.git import git_commit
from skore.sklearn.types import DataSource, ReportMetadata
from skore.utils._progress_bar import track
from skore.utils._uuid import normalize_report_id, uuid7
from skore.utils.docscrape import (
    build_numpy_docstring,
    callable_docstring,
    docstring_summary,
    parameters_by_name,
    parse_numpy_doc,
    replace_default,
)
from skore.utils.repr.base import (
    AccessorHelpMixin,
    ReportHelpMixin,
    render_panel_to_plain_text,
)
from skore.utils.repr.data import MethodHelp

if TYPE_CHECKING:
    import pandas as pd

    from skore.checks.accessor import _ChecksAccessor
    from skore.displays import MetricsSummaryDisplay
    from skore.reports.cross_validation.report import CrossValidationReport
    from skore.reports.estimator.report import EstimatorReport


class _BaseReport(ReportHelpMixin):
    """Base class for all reports.

    This class centralizes shared report logic (e.g. configuration, accessors) and
    inherits from ``ReportHelpMixin`` to provide a consistent ``help()`` and rich/HTML
    representation across all report types.
    """

    _ACCESSOR_CONFIG: dict[str, dict[str, str]]
    _report_type: Literal[
        "estimator",
        "cross-validation",
        "comparison-estimator",
        "comparison-cross-validation",
    ]

    checks: _ChecksAccessor

    def _aggregate_checks(
        self, ignored_codes: set[CheckCode], *, fast_mode: bool = False
    ) -> dict[CheckCode, CheckResult]:
        """Aggregate EstimatorReport checks.

        Overwritten in Comparison reports.
        """
        return {}

    def _get_checks_results(
        self, ignored_codes: set[CheckCode], *, fast_mode: bool = False
    ) -> dict[CheckCode, CheckResult]:
        """Run uncached checks and return the checks summary.

        Parameters
        ----------
        ignored_codes : set of CheckCode
            Check codes to exclude from execution, e.g. ``{"SKD001"}``.

        fast_mode : bool, default=False
            When True, skip slow checks that are not already in the cache
            (their `check_function` is never invoked). Cached slow results
            are still surfaced.

        Returns
        -------
        dict of CheckCode to CheckResult
            Summary of every check applicable to the report type with its display
            section.
        """
        if not hasattr(self, "_check_results_cache"):
            self._check_results_cache: dict[CheckCode, CheckResult] = {}

        checks_to_run = [
            check
            for check in self._checks_registry
            if self._report_type in check.report_types
            and check.code not in self._check_results_cache
            and check.code not in ignored_codes
            and not (fast_mode and check.slow)
        ]
        for check in track(
            checks_to_run,
            description="Running checks",
            total=len(checks_to_run),
            disable=fast_mode,
        ):
            try:
                explanation = check.check_function(self)
                section: CheckSection = (
                    getattr(check, "severity", "issue") if explanation else "passed"
                )
            except CheckNotApplicable as exc:
                explanation = exc.args[0] if exc.args else None
                section = "not_applicable"
            self._check_results_cache[check.code] = {
                "title": check.title,
                "docs_url": check.docs_url,
                "explanation": explanation,
                "section": section,
            }

        if "comparison" in self._report_type:
            return self._aggregate_checks(ignored_codes, fast_mode=fast_mode)

        summary: dict[CheckCode, CheckResult] = {}
        # NOTE: Every check should appear exactly once in the summary
        for check in self._checks_registry:
            if self._report_type not in check.report_types:
                summary[check.code] = {
                    "title": check.title,
                    "docs_url": check.docs_url,
                    "explanation": f"Not applicable to {self._report_type} reports.",
                    "section": "not_applicable",
                }
            elif check.code in ignored_codes:
                summary[check.code] = {
                    "title": check.title,
                    "docs_url": check.docs_url,
                    "explanation": None,
                    "section": "ignored",
                }
            elif (
                fast_mode
                and check.slow
                and (check.code not in self._check_results_cache)
            ):
                summary[check.code] = {
                    "title": check.title,
                    "docs_url": check.docs_url,
                    "explanation": None,
                    "section": "skipped",
                }
            elif check.code in self._check_results_cache:
                summary[check.code] = self._check_results_cache[check.code]

        return summary

    def _checks_summary_html_fragment(self) -> str:
        """HTML snippet for the checks summary tab in report reprs."""
        return self.checks.summarize(fast_mode=True)._embedded_repr_html()

    @staticmethod
    def _normalize_metadata(metadata: dict[str, Any]) -> ReportMetadata:
        normalized = metadata.copy()
        normalized["id"] = normalize_report_id(metadata["id"])
        return cast(ReportMetadata, normalized)

    def __init__(self) -> None:
        self._checks_registry: list[Check] = list(_BUILTIN_CHECKS)
        self._metadata: ReportMetadata = {
            "id": str(uuid7()),
            "skore-version": version("skore"),
            "creation-date": datetime.now(UTC).isoformat(),
            # comparison reports don't have a _report_type yet at init time
            # but they don't have a `to_dict` anyway:
            "report_type": getattr(self, "_report_type", "comparison"),
            "git_commit": git_commit(),
        }

    @property
    def id(self) -> str:
        return self._metadata["id"]

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if "_metadata" in state:
            self._metadata = self._normalize_metadata(state["_metadata"])
        elif "id" in state:
            self._metadata = {
                "id": normalize_report_id(state["id"]),
                "skore-version": "legacy",
                "creation-date": "",
                "report_type": getattr(self, "_report_type", "comparison"),
                "git_commit": None,
            }


ParentT = TypeVar("ParentT", bound="_BaseReport")


class _BaseAccessor(AccessorHelpMixin, Generic[ParentT]):
    """Base class for all accessors.

    Accessors expose additional views on a report (e.g. data, metrics) and inherit from
    ``AccessorHelpMixin`` to provide a dedicated ``help()`` and rich/HTML help tree.
    """

    def __init__(self, parent: ParentT) -> None:
        self._parent = parent

    def __repr__(self) -> str:
        return render_panel_to_plain_text(self._create_help_panel())

    def _repr_html_(self) -> str:
        return self._create_help_html()

    def _repr_mimebundle_(self, **kwargs):
        return {"text/plain": repr(self), "text/html": self._repr_html_()}


def _summarize_report_metrics(
    report: EstimatorReport | CrossValidationReport,
    *,
    data_source: DataSource | Literal["both"],
    metric: str | list[str] | None = None,
) -> MetricsSummaryDisplay:
    """Compute a metrics summary for ``report``.

    Pure Python function to avoid pickling a metrics accessor as a bound method's
    ``__self__``.
    """
    return report.metrics._summarize_display(data_source=data_source, metric=metric)


class BaseMetricsAccessor(_BaseAccessor, Generic[ParentT]):
    """Base class for metrics accessor."""

    # Help tree subgroups under ``.metrics``. Registry callables are injected
    # into Metrics by ``_build_help_data``.
    _HELP_METHOD_GROUPS: ClassVar[dict[str, tuple[str, ...]]] = {
        "Registry": ("available", "add", "remove", "get"),
        "Metrics": ("fit_time", "predict_time", "score", "timings"),
        "Displays": (
            "summarize",
            "roc",
            "precision_recall",
            "prediction_error",
            "confusion_matrix",
        ),
    }

    def _callable_metric_names(self) -> list[str]:
        """Registry metric names that can be exposed as ``metrics.<name>()``."""
        return [
            name
            for name in self.available()
            if name.isidentifier()
            and not iskeyword(name)
            and not hasattr(type(self), name)
        ]

    @abstractmethod
    def _resolve_metric(self, name: str) -> Metric | None:
        """Return the :class:`~skore.metrics.Metric` for ``name``, or None."""

    def _metric_summary(self, metric: Metric) -> str:
        """Summarize ``metric`` in a single sentence."""
        summary = docstring_summary(callable_docstring(metric.function))
        if not summary:
            # Prefer a class docstring the metric owns. Plain ``Metric`` instances
            # (from ``Metric.new``) inherit the base class docstring, which
            # describes the machinery rather than the metric itself.
            class_doc = type(metric).__doc__
            if class_doc is not None and class_doc is not Metric.__doc__:
                summary = docstring_summary(class_doc)
        return summary or metric.verbose_name or "Registered metric."

    def _build_metric_method_docstring(self, name: str) -> str:
        """Build a numpydoc string for a dynamically exposed registry metric."""
        metric = self._resolve_metric(name)

        get_doc = getattr(self.get, "__doc__", None)
        get_parsed = parse_numpy_doc(get_doc)
        get_params = parameters_by_name(get_parsed)

        parameters: list[Parameter] = []
        shared_names: set[str] = set()
        for param in inspect.signature(self.get).parameters.values():
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

        if metric is None:
            summary = "Registered metric."
        else:
            summary = self._metric_summary(metric)
            score_params = parameters_by_name(callable_docstring(metric.function))
            for key, default in metric.kwargs.items():
                if key in shared_names:
                    continue
                score_param = score_params.get(key)
                if score_param is not None and score_param.type:
                    type_spec = replace_default(score_param.type, default)
                else:
                    type_spec = f"default={default!r}"
                description = (
                    list(score_param.desc)
                    if score_param is not None and score_param.desc
                    else ["Forwarded to the underlying score function."]
                )
                parameters.append(Parameter(key, type_spec, description))

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

    def _metric_help_description(self, name: str) -> str:
        """Build a help description for a registry metric method."""
        metric = self._resolve_metric(name)
        if metric is None:
            return "Registered metric."
        return self._metric_summary(metric)

    def __getattr__(self, name):
        """Expose registry metrics as methods when not defined statically.

        If attribute ``name`` is defined statically, this method will not be called.
        """
        if name in self._callable_metric_names():
            method = partial(self.get, name)
            method.__doc__ = self._build_metric_method_docstring(name)
            method.__name__ = name
            method.__qualname__ = f"{type(self).__qualname__}.{name}"
            return method

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __dir__(self) -> list[str]:
        """Add registry metrics to ``__dir__`` for tab-completion."""
        return list(set(super().__dir__()).union(self._callable_metric_names()))

    def _extra_help_methods(self) -> list[MethodHelp]:
        """Help entries for the registry metrics exposed through ``__getattr__``.

        Registry metrics are only reachable through ``__getattr__``, so they are not
        picked up by the default method-discovery logic used to build help data.
        Names that are not valid identifiers (e.g. containing spaces) are excluded,
        since they cannot be called as ``report.metrics.<name>(...)``.
        """
        # Registry metrics have no Sphinx API page; show the summary tooltip only.
        # Keep registry order (``available()``) instead of sorting alphabetically.
        return [
            MethodHelp(
                name=name,
                parameters="(...)",
                description=self._metric_help_description(name),
            )
            for name in self._callable_metric_names()
        ]

    def _help_method_group_spec(self) -> dict[str, tuple[str, ...]]:
        """Partition methods into Registry / Metrics / Displays groups.

        Registry callables are listed first in Metrics, ahead of the static score
        helpers declared on the class. Some helpers (e.g. ``fit_time``) are registry
        metrics on reports that do not define them as methods, hence the dedupe.
        """
        group_spec = dict(self._HELP_METHOD_GROUPS)
        registry_names = tuple(self._callable_metric_names())
        group_spec["Metrics"] = registry_names + tuple(
            name for name in group_spec["Metrics"] if name not in set(registry_names)
        )
        return group_spec

    def _formatted_summary_frame(
        self,
        *,
        data_source: DataSource = "test",
        metric: str | list[str] | None = None,
    ) -> pd.DataFrame | pd.Series:
        """Metric summary frame used for accessor display."""
        display = self.summarize(data_source=data_source, metric=metric)
        flat_index = "comparison" not in display.report_type
        return display.frame(flat_index=flat_index)

    def __repr__(self) -> str:
        return (
            "Metrics summary:\n"
            f"{self._formatted_summary_frame()!r}\n"
            "Explore available methods with .help()."
        )

    def _repr_html_(self) -> str:
        frame = self.summarize().frame(verbose_name=True, flat_index=False)
        html = (
            frame.to_frame()._repr_html_()
            if isinstance(frame, pd.Series)
            else frame._repr_html_()
        )
        return (
            "<p>Metrics summary:</p>"
            f"{html}"
            '<p role="note">Explore available methods with '
            "<code>.help()</code>.</p>"
        )
