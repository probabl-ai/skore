"""Experiment collection stored beside a project's reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pandas import DataFrame

from skore._project.types import ExperimentRow, ExperimentStatus

if TYPE_CHECKING:
    from skore import CrossValidationReport, EstimatorReport
    from skore._project.project import Project
    from skore._project.types import OnSameStatus

EXPERIMENT_COLUMNS = (
    "id",
    "stem",
    "status",
    "question",
    "plan",
    "source_id",
    "check_code",
    "abandon_reason",
    "version",
    "created_at",
    "updated_at",
)
STATUSES = frozenset(
    {"aside", "discarded", "backlog", "running", "history", "abandoned"}
)
TRIAGE_STATUSES = frozenset({"aside", "discarded"})
PLAN_STATUSES = frozenset({"aside", "backlog", "running"})
RUN_STATUSES = frozenset({"backlog", "running"})
_TRANSITIONS = frozenset(
    {
        ("aside", "discarded"),
        ("discarded", "aside"),
        ("aside", "backlog"),
        ("backlog", "running"),
        ("running", "backlog"),
        ("running", "abandoned"),
        ("backlog", "history"),
        ("running", "history"),
    }
)


def _none_if_blank(value: str | None) -> str | None:
    """Return stripped text, or ``None`` when ``value`` is missing or blank.

    ``None``, an empty string, and whitespace-only text are the same missing
    value.
    """
    if value is None:
        return None
    return value.strip() or None


def _normalize_row(row: ExperimentRow | dict[str, Any]) -> ExperimentRow:
    """Return the row in stored form, with text stripped and blanks as None."""
    status = row["status"]
    if status not in STATUSES:
        raise ValueError(
            f"Unknown experiment status {status!r}. Expected one of {sorted(STATUSES)}."
        )
    stem = str(row["stem"]).strip()
    question = str(row["question"]).strip()
    if not stem:
        raise ValueError("An experiment stem is required.")
    if not question:
        raise ValueError("An experiment question is required.")
    reason = _none_if_blank(row.get("abandon_reason"))
    if status == "abandoned" and reason is None:
        raise ValueError("abandon() requires a reason.")
    return {
        "id": str(row["id"]),
        "stem": stem,
        "status": status,
        "question": question,
        "plan": _none_if_blank(row.get("plan")),
        "source_id": _none_if_blank(row.get("source_id")),
        "check_code": _none_if_blank(row.get("check_code")),
        "abandon_reason": reason,
        "version": int(row["version"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }


def _transition_allowed(current: ExperimentStatus, new: ExperimentStatus) -> bool:
    """Return whether a row in ``current`` may move to ``new``."""
    if current == new:
        return current in PLAN_STATUSES
    return (current, new) in _TRANSITIONS


class _WriteConflict(Exception):
    """A backend rejected an experiment write. ``current`` is the stored row."""

    def __init__(self, message: str, current: ExperimentRow) -> None:
        super().__init__(message)
        self.current = current


class JournalConflictError(Exception):
    """A journal write was based on an outdated record.

    Attributes
    ----------
    current : Experiment or None
        The record stored now, when one exists.
    """

    def __init__(self, message: str, *, current: Experiment | None = None) -> None:
        super().__init__(message)
        self.current = current


class Experiment:
    """One experiment stored in a project.

    Callers do not construct this object. :func:`skore.agent.create_experiment`
    appends a row and returns the handle. :meth:`Experiments.get` returns one
    as well. Each write returns a new handle; the previous one is stale.

    ``status`` is one of ``"aside"``, ``"discarded"``, ``"backlog"``,
    ``"running"``, ``"history"``, and ``"abandoned"``.
    """

    def __init__(self, project: Project, row: ExperimentRow) -> None:
        self._project = project
        self._row = row

    @property
    def id(self) -> str:
        """The UUID7 of this experiment.

        Once :meth:`run` stores a report, that report carries the same id.
        """
        return self._row["id"]

    @property
    def stem(self) -> str:
        """The key under which :meth:`run` stores the report."""
        return self._row["stem"]

    @property
    def status(self) -> ExperimentStatus:
        """Where this experiment sits."""
        return self._row["status"]

    @property
    def question(self) -> str:
        """The question this experiment is meant to answer."""
        return self._row["question"]

    @property
    def plan(self) -> str | None:
        """How to run this experiment, or ``None`` when it has no plan yet."""
        return self._row["plan"]

    @property
    def source_id(self) -> str | None:
        """The report or experiment id that raised this row, when one did."""
        return self._row["source_id"]

    @property
    def check_code(self) -> str | None:
        """The check code that raised this row, when one did."""
        return self._row["check_code"]

    @property
    def abandon_reason(self) -> str | None:
        """Why this experiment was abandoned, or ``None`` otherwise."""
        return self._row["abandon_reason"]

    @property
    def version(self) -> int:
        """The version this handle was read from. Writes send it back."""
        return self._row["version"]

    @property
    def created_at(self) -> str:
        """When this experiment was created, as an ISO 8601 timestamp."""
        return self._row["created_at"]

    @property
    def updated_at(self) -> str:
        """When this experiment was last written, as an ISO 8601 timestamp."""
        return self._row["updated_at"]

    def decide(self, status: ExperimentStatus) -> Experiment:
        """Move this experiment between ``aside`` and ``discarded``.

        Moving a row to the status it already has does not write.

        Parameters
        ----------
        status : {"aside", "discarded"}
            The triage status to store.

        Returns
        -------
        experiment : Experiment
            The handle after the call.
        """
        if status not in TRIAGE_STATUSES or self.status not in TRIAGE_STATUSES:
            raise ValueError(
                f"Cannot decide an experiment from {self.status!r} to {status!r}. "
                "decide() only moves between 'aside' and 'discarded'."
            )
        return self._write({**self._row, "status": status}, on_same_status="noop")

    def promote(self, plan: str | None = None) -> Experiment:
        """Move an ``aside`` experiment to the backlog.

        ``plan`` is optional. Promoting a row that is already in the backlog
        does not write, and a plan passed to that call is ignored.

        Parameters
        ----------
        plan : str or None, default=None
            How the experiment will be run. Replaces the stored plan when given.

        Returns
        -------
        experiment : Experiment
            The handle after the call.
        """
        if self.status not in {"aside", "backlog"}:
            raise ValueError(
                f"Cannot promote an experiment from {self.status!r}. "
                "Promote an experiment that is 'aside'."
            )
        row = {**self._row, "status": "backlog"}
        if plan is not None:
            text = _none_if_blank(plan)
            if text is None:
                raise ValueError("promote() does not accept a blank plan.")
            row["plan"] = text
        return self._write(row, on_same_status="noop")

    def update_plan(self, plan: str) -> Experiment:
        """Replace the plan while the experiment is not finished.

        The plan can change in ``aside``, ``backlog``, and ``running``. It is
        frozen once the experiment is ``history`` or ``abandoned``.

        Parameters
        ----------
        plan : str
            The new plan. A blank plan is rejected.

        Returns
        -------
        experiment : Experiment
            The handle after the plan is stored.
        """
        text = _none_if_blank(plan)
        if text is None:
            raise ValueError("A plan cannot be blank.")
        if self.status not in PLAN_STATUSES:
            raise ValueError(
                f"Cannot update the plan of an experiment in status {self.status!r}."
            )
        return self._write({**self._row, "plan": text})

    def start(self) -> Experiment:
        """Move a backlog experiment to ``running``.

        A second start, including one whose handle already says ``running``,
        raises :class:`JournalConflictError`.

        Returns
        -------
        experiment : Experiment
            The handle after the row is running.
        """
        if self.status not in {"backlog", "running"}:
            raise ValueError(
                f"Cannot start an experiment from {self.status!r}. "
                "start() requires the backlog."
            )
        return self._write(
            {**self._row, "status": "running"}, on_same_status="conflict"
        )

    def release(self) -> Experiment:
        """Move a running experiment back to the backlog.

        Returns
        -------
        experiment : Experiment
            The handle after the row is in the backlog.
        """
        if self.status != "running":
            raise ValueError(
                f"Cannot release an experiment from {self.status!r}. "
                "release() requires a running experiment."
            )
        return self._write({**self._row, "status": "backlog"})

    def abandon(self, reason: str) -> Experiment:
        """Move a running experiment to ``abandoned`` and store ``reason``.

        Parameters
        ----------
        reason : str
            Why the experiment was dropped. A blank reason is rejected.

        Returns
        -------
        experiment : Experiment
            The handle after the row is abandoned.
        """
        text = _none_if_blank(reason)
        if text is None:
            raise ValueError("abandon() requires a reason.")
        if self.status != "running":
            raise ValueError(
                f"Cannot abandon an experiment from {self.status!r}. "
                "abandon() requires a running experiment."
            )
        return self._write({**self._row, "status": "abandoned", "abandon_reason": text})

    def run(
        self,
        estimator: Any,
        X: Any = None,
        y: Any = None,
        **evaluate_kwargs: Any,
    ) -> EstimatorReport | CrossValidationReport:
        """Evaluate ``estimator`` and store the report under this experiment's id.

        From the backlog, the row is moved to ``running`` first. A failure of
        evaluation or storage then moves that row back to the backlog. A row
        that was already running stays running when evaluation or storage
        fails. On success the row moves to ``history``.

        Parameters
        ----------
        estimator : estimator
            The scikit-learn compatible estimator to evaluate.
        X : array-like, default=None
            The data to evaluate on.
        y : array-like, default=None
            The target.
        **evaluate_kwargs : dict
            Extra keyword arguments passed to :func:`~skore.evaluate`.

        Returns
        -------
        report : EstimatorReport or CrossValidationReport
            The report stored in the project. Its id is this experiment's id.
        """
        if self.status not in RUN_STATUSES:
            raise ValueError(
                f"Cannot run an experiment from {self.status!r}. "
                "run() requires the backlog or a running experiment."
            )
        current = self.start() if self.status == "backlog" else self
        try:
            from skore import CrossValidationReport, EstimatorReport
            from skore._dispatch.evaluate import evaluate

            report = evaluate(estimator, X, y, **evaluate_kwargs)
            if not isinstance(report, EstimatorReport | CrossValidationReport):
                raise TypeError("run() stores an estimator or cross-validation report.")
            report._metadata["id"] = current.id
            current._project.put(current.stem, report)
        except Exception:
            if self.status == "backlog":
                current.release()
            raise
        current._project._update_experiment({**current._row, "status": "history"})
        return report

    def _write(
        self, row: dict[str, Any], *, on_same_status: OnSameStatus = "write"
    ) -> Experiment:
        """Store ``row`` and return the handle read back from the project."""
        return self._project._update_experiment(row, on_same_status=on_same_status)


def _as_conflict(error: _WriteConflict, project: Project) -> JournalConflictError:
    """Return the public conflict error for a rejected backend write."""
    return JournalConflictError(
        str(error), current=Experiment(project, _normalize_row(error.current))
    )


class Experiments:
    """The experiments stored in one project.

    Obtain this collection from :attr:`Project.experiments`. Every read asks
    the backend again.
    """

    def __init__(self, project: Project) -> None:
        self._project = project

    @property
    def frame(self) -> DataFrame:
        """The stored columns, one row per experiment, ordered by creation time.

        The columns are ``id``, ``stem``, ``status``, ``question``, ``plan``,
        ``source_id``, ``check_code``, ``abandon_reason``, ``version``,
        ``created_at``, and ``updated_at``. An empty project returns a frame
        with those columns and no rows.

        Returns
        -------
        frame : pandas.DataFrame
            One row per experiment.
        """
        rows = sorted(
            self._project._experiment_rows(),
            key=lambda row: (row["created_at"], row["id"]),
        )
        if not rows:
            return DataFrame(columns=list(EXPERIMENT_COLUMNS))
        return DataFrame.from_records(rows, columns=list(EXPERIMENT_COLUMNS))

    def get(self, experiment_id: str) -> Experiment:
        """Return the handle for ``experiment_id``.

        Parameters
        ----------
        experiment_id : str
            The ``id`` of a stored experiment.

        Returns
        -------
        experiment : Experiment
            The handle for that row.
        """
        for row in self._project._experiment_rows():
            if row["id"] == str(experiment_id):
                return Experiment(self._project, row)
        raise KeyError(experiment_id)
