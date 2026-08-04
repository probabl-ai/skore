from __future__ import annotations

import inspect
from abc import abstractmethod
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import version
from keyword import iskeyword
from typing import TYPE_CHECKING, ClassVar, Generic, Literal, TypeVar
from uuid import uuid4

import pandas as pd

from skore._externals._docscrape import Parameter
from skore._project.git import git_commit
from skore._sklearn._checks._utils import CheckNotApplicable
from skore._sklearn._checks.base import Check, CheckCode, CheckResult, CheckSection
from skore._sklearn._checks.model_checks import _BUILTIN_CHECKS
from skore._sklearn.metrics import Metric
from skore._sklearn.types import DataSource, ReportMetadata
from skore._utils._progress_bar import track
from skore._utils.docscrape import (
    build_numpy_docstring,
    callable_docstring,
    docstring_summary,
    parameters_by_name,
    parse_numpy_doc,
    replace_default,
)
from skore._utils.repr.base import (
    AccessorHelpMixin,
    ReportHelpMixin,
    render_panel_to_plain_text,
)
from skore._utils.repr.data import MethodGroupHelp, MethodHelp

if TYPE_CHECKING:
    import pandas as pd

    from skore._sklearn._checks.accessor import _ChecksAccessor
    from skore._sklearn._cross_validation.report import CrossValidationReport
    from skore._sklearn._estimator.report import EstimatorReport
    from skore._sklearn._plot import MetricsSummaryDisplay
    from skore._utils.repr.data import AccessorHelpData


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
        for check in self._checks_registry:
            if self._report_type not in check.report_types:
                continue
            code = check.code
            if code in ignored_codes:
                summary[code] = {
                    "title": check.title,
                    "docs_url": check.docs_url,
                    "explanation": None,
                    "section": "ignored",
                }
            elif fast_mode and check.slow and code not in self._check_results_cache:
                summary[code] = {
                    "title": check.title,
                    "docs_url": check.docs_url,
                    "explanation": None,
                    "section": "skipped",
                }
            elif code in self._check_results_cache:
                summary[code] = self._check_results_cache[code]
        return summary

    def _checks_summary_html_fragment(self) -> str:
        """HTML snippet for the checks summary tab in report reprs."""
        return self.checks.summarize(fast_mode=True)._embedded_repr_html()

    def __init__(self) -> None:
        self._checks_registry: list[Check] = list(_BUILTIN_CHECKS)
        self._metadata: ReportMetadata = {
            "id": uuid4().int,
            "skore-version": version("skore"),
            "creation-date": datetime.now(UTC).isoformat(),
            # comparison reports don't have a _report_type yet at init time
            # but they don't have a `to_dict` anyway:
            "report_type": getattr(self, "_report_type", "comparison"),
            "git_commit": git_commit(),
        }

    @property
    def id(self) -> int:
        return self._metadata["id"]


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

    # Help tree subgroups under ``.metrics``. ``Displays`` is a catch-all for
    # methods that are neither registry management nor metric callables.
    _HELP_METHOD_GROUPS: ClassVar[dict[str, tuple[str, ...] | None]] = {
        "Registry": ("available", "add", "remove", "get"),
        "Metrics": ("fit_time", "predict_time", "score", "timings"),
        "Displays": None,
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
        """Return the :class:`~skore._sklearn.metrics.Metric` for ``name``, or None."""

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

    def _build_help_data(self) -> AccessorHelpData:
        """Include registry metrics in the help data.

        Registry metrics are only reachable through ``__getattr__``, so they are not
        picked up by the default method-discovery logic used to build help data.
        Names that are not valid identifiers (e.g. containing spaces) are excluded,
        since they cannot be called as ``report.metrics.<name>(...)``.

        Methods are then partitioned into Registry / Metrics / Displays groups.
        Metrics combines registry callables with static score helpers; Displays
        collects remaining methods that are neither registry management nor metrics.
        """
        help_data = super()._build_help_data()
        known_names = {method.name for method in help_data.methods}
        # Registry metrics have no Sphinx API page; show the summary tooltip only.
        # Keep registry order (``available()``) instead of sorting alphabetically.
        help_data.methods.extend(
            MethodHelp(
                name=name,
                parameters="(...)",
                description=self._metric_help_description(name),
            )
            for name in self._callable_metric_names()
            if name not in known_names
        )

        by_name = {method.name: method for method in help_data.methods}
        registry_names = self._HELP_METHOD_GROUPS["Registry"] or ()
        static_metric_names = self._HELP_METHOD_GROUPS["Metrics"] or ()
        registry_order = self._callable_metric_names()

        registry_methods = [by_name[name] for name in registry_names if name in by_name]
        metrics_methods = [
            by_name[name] for name in registry_order if name in by_name
        ] + [by_name[name] for name in static_metric_names if name in by_name]
        claimed = {method.name for method in registry_methods + metrics_methods}
        displays_methods = [
            method for method in help_data.methods if method.name not in claimed
        ]

        groups: list[MethodGroupHelp] = []
        if registry_methods:
            groups.append(
                MethodGroupHelp(
                    branch_id=str(uuid4()),
                    name="Registry",
                    methods=registry_methods,
                )
            )
        if metrics_methods:
            groups.append(
                MethodGroupHelp(
                    branch_id=str(uuid4()),
                    name="Metrics",
                    methods=metrics_methods,
                )
            )
        if displays_methods:
            groups.append(
                MethodGroupHelp(
                    branch_id=str(uuid4()),
                    name="Displays",
                    methods=displays_methods,
                )
            )
        help_data.groups = groups or None
        return help_data

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
