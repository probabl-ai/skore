"""Experiment collection stored beside a project's reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pandas import DataFrame

from skore._project.types import ExperimentRow, ExperimentStatus
from skore._utils.uuid import uuid7

if TYPE_CHECKING:
    from skore import CrossValidationReport, EstimatorReport
    from skore._project.project import Project


EXPERIMENT_COLUMNS = (
    "id",
    "stem",
    "status",
    "question",
    "source_id",
    "check_code",
    "plan",
)
STATUSES = frozenset({"idea", "aside", "discarded", "backlog", "history"})
TRIAGE_STATUSES = frozenset({"idea", "aside", "discarded"})


def _blank(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_row(row: dict[str, Any]) -> ExperimentRow:
    """Return a stored experiment row with only the seven persisted columns."""
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
    return {
        "id": str(row["id"]),
        "stem": stem,
        "status": status,
        "question": question,
        "source_id": _blank(row.get("source_id")),
        "check_code": _blank(row.get("check_code")),
        "plan": _blank(row.get("plan")),
    }


def transition_allowed(
    current: ExperimentStatus | None,
    new: ExperimentStatus,
    plan: str | None,
) -> bool:
    """Return whether ``current`` may become ``new``.

    A missing current row is an insert, which sync uses to copy a row that
    already reached ``history`` on another backend. An existing row only moves
    along decide, promote, and run.
    """
    if new not in STATUSES:
        return False
    if current is None or current == new:
        return True
    if current == "history":
        return False
    if current in TRIAGE_STATUSES and new in TRIAGE_STATUSES:
        return True
    if current == "idea" and new == "backlog":
        return plan is not None
    return current == "backlog" and new == "history"


def ensure_report_id(report: Any, report_id: str | None = None) -> str:
    """Assign a UUID7 when ``report`` has none, and keep an existing id.

    Nested estimator reports of a cross-validation report receive their own
    ids. Only the report passed by the caller receives ``report_id``.
    """
    current = report._metadata.get("id")
    if current:
        if report_id is not None and current != report_id:
            raise ValueError(
                f"Report id {current!r} does not match experiment id {report_id!r}."
            )
        assigned = current
    else:
        assigned = report_id or str(uuid7())
        report._metadata["id"] = assigned
    for child in getattr(report, "reports_", ()) or ():
        if child is not report:
            ensure_report_id(child)
    return assigned


class Experiment:
    """One experiment stored in a project.

    An experiment is one row of :attr:`Experiments.frame`. Callers do not
    construct this object. :func:`skore.agent.create_experiment` appends a row
    and returns the handle. :meth:`Experiments.get`, :meth:`decide`, and
    :meth:`promote` return one as well.

    ``status`` is one of ``"idea"``, ``"aside"``, ``"discarded"``,
    ``"backlog"``, and ``"history"``.

    See Also
    --------
    Experiments :
        The collection that lists every experiment in a project.
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
        """Where this experiment sits.

        One of ``"idea"``, ``"aside"``, ``"discarded"``, ``"backlog"``, and
        ``"history"``.
        """
        return self._row["status"]

    @property
    def question(self) -> str:
        """The question this experiment is meant to answer."""
        return self._row["question"]

    @property
    def plan(self) -> str | None:
        """How to run this experiment, or ``None`` while it is still an idea."""
        return self._row["plan"]

    @property
    def source_id(self) -> str | None:
        """The report id that raised this experiment, when one did."""
        return self._row["source_id"]

    @property
    def check_code(self) -> str | None:
        """The check code that raised this experiment, when one did."""
        return self._row["check_code"]

    def decide(self, status: ExperimentStatus) -> Experiment:
        """Move this row among ``idea``, ``aside``, and ``discarded``.

        Parameters
        ----------
        status : {"idea", "aside", "discarded"}
            The triage status to store.

        Returns
        -------
        experiment : Experiment
            The handle after the row is stored.

        Raises
        ------
        ValueError
            If this experiment is not in triage, or ``status`` is not a triage
            status.
        """
        if self.status not in TRIAGE_STATUSES or status not in TRIAGE_STATUSES:
            raise ValueError(
                f"Cannot decide an experiment from {self.status!r} to {status!r}. "
                "decide() only moves among 'idea', 'aside', and 'discarded'."
            )
        return self._project._write_experiment({**self._row, "status": status})

    def promote(self, plan: str) -> Experiment:
        """Move an idea to the backlog and store its plan.

        Parameters
        ----------
        plan : str
            How the experiment will be run. A blank plan is rejected.

        Returns
        -------
        experiment : Experiment
            The handle after the row is stored in the backlog.

        Raises
        ------
        ValueError
            If ``plan`` is blank, or this experiment is not an idea.
        """
        text = _blank(plan)
        if text is None:
            raise ValueError("promote() requires a plan.")
        if self.status != "idea":
            raise ValueError(
                f"Cannot promote an experiment from {self.status!r}. "
                "Decide it back to 'idea' first."
            )
        return self._project._write_experiment(
            {**self._row, "status": "backlog", "plan": text}
        )

    def run(
        self,
        estimator: Any,
        X: Any = None,
        y: Any = None,
        **evaluate_kwargs: Any,
    ) -> EstimatorReport | CrossValidationReport:
        """Evaluate ``estimator`` and store the report under this experiment's id.

        The experiment must be in the backlog, and it can be run only once. The
        stored report receives this experiment's :attr:`id`.

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
            The report stored in the project.

        Raises
        ------
        ValueError
            If this experiment is not in the backlog, or it was already run.
        TypeError
            If evaluation does not return an estimator or cross-validation
            report.
        """
        if self.status == "history":
            raise ValueError("An experiment can be run only once.")
        if self.status != "backlog":
            raise ValueError("run() requires an experiment in the backlog.")
        try:
            self._project.get(self.id)
        except Exception:
            pass
        else:
            raise ValueError("An experiment can be run only once.")

        from skore import CrossValidationReport, EstimatorReport, evaluate

        report = evaluate(estimator, X, y, **evaluate_kwargs)
        if not isinstance(report, EstimatorReport | CrossValidationReport):
            raise TypeError("run() stores an estimator or cross-validation report.")
        ensure_report_id(report, self.id)
        stored = self._project._store_report(self.stem, report)
        self._project._write_experiment({**self._row, "status": "history"})
        self._project._raise_check_ideas(self.stem, stored)
        return stored


class Experiments:
    """The experiments stored in one project.

    Obtain this collection from :attr:`Project.experiments`. Do not construct
    it directly.

    See Also
    --------
    Experiment :
        One row and the methods that move it.
    """

    def __init__(self, project: Project) -> None:
        self._project = project

    @property
    def frame(self) -> DataFrame:
        """The stored columns, one row per experiment.

        The columns are ``id``, ``stem``, ``status``, ``question``,
        ``source_id``, ``check_code``, and ``plan``. An empty project returns
        a frame with those columns and no rows.

        Returns
        -------
        frame : pandas.DataFrame
            One row per experiment.
        """
        rows = self._project._experiment_rows()
        if not rows:
            return DataFrame(columns=list(EXPERIMENT_COLUMNS))
        frame = DataFrame.from_records(
            [normalize_row(row) for row in rows], columns=list(EXPERIMENT_COLUMNS)
        )
        return frame

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

        Raises
        ------
        KeyError
            If no experiment has this id.
        """
        matched = self.frame.loc[self.frame["id"].astype(str) == str(experiment_id)]
        if matched.empty:
            raise KeyError(experiment_id)
        row = normalize_row(matched.iloc[0].to_dict())
        return Experiment(self._project, row)
