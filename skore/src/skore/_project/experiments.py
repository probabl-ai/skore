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
    """Handle on one row of ``project.experiments.frame``."""

    def __init__(self, project: Project, row: ExperimentRow) -> None:
        self._project = project
        self._row = row

    @property
    def id(self) -> str:
        return self._row["id"]

    @property
    def stem(self) -> str:
        return self._row["stem"]

    @property
    def status(self) -> ExperimentStatus:
        return self._row["status"]

    @property
    def question(self) -> str:
        return self._row["question"]

    @property
    def plan(self) -> str | None:
        return self._row["plan"]

    @property
    def source_id(self) -> str | None:
        return self._row["source_id"]

    @property
    def check_code(self) -> str | None:
        return self._row["check_code"]

    def decide(self, status: ExperimentStatus) -> Experiment:
        """Move this row among ``idea``, ``aside``, and ``discarded``."""
        if self.status not in TRIAGE_STATUSES or status not in TRIAGE_STATUSES:
            raise ValueError(
                f"Cannot decide an experiment from {self.status!r} to {status!r}. "
                "decide() only moves among 'idea', 'aside', and 'discarded'."
            )
        return self._project._write_experiment({**self._row, "status": status})

    def promote(self, plan: str) -> Experiment:
        """Move an idea to the backlog and store its plan."""
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
        """Evaluate ``estimator`` and store the report under this experiment's id."""
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
    """The experiments stored in one project."""

    def __init__(self, project: Project) -> None:
        self._project = project

    @property
    def frame(self) -> DataFrame:
        """The stored columns, one row per experiment."""
        rows = self._project._experiment_rows()
        if not rows:
            return DataFrame(columns=list(EXPERIMENT_COLUMNS))
        frame = DataFrame.from_records(
            [normalize_row(row) for row in rows], columns=list(EXPERIMENT_COLUMNS)
        )
        return frame

    def get(self, experiment_id: str) -> Experiment:
        """Return the handle for ``experiment_id``."""
        matched = self.frame.loc[self.frame["id"].astype(str) == str(experiment_id)]
        if matched.empty:
            raise KeyError(experiment_id)
        row = normalize_row(matched.iloc[0].to_dict())
        return Experiment(self._project, row)
