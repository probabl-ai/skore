"""Public entry point for creating experiments."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from skore._project.experiments import _normalize_row
from skore._utils.uuid import uuid7

if TYPE_CHECKING:
    from skore._project.experiments import Experiment
    from skore._project.project import Project

__all__ = ["create_experiment"]


def create_experiment(
    project: Project,
    stem: str,
    question: str,
    *,
    status: Literal["aside", "backlog"] = "aside",
    plan: str | None = None,
    source_id: str | None = None,
    check_code: str | None = None,
) -> Experiment:
    """Append one experiment and return a handle on it.

    New rows start as ``"aside"`` or, when ``status="backlog"``, go straight
    to the backlog. The returned id is the UUID7 the report will carry when
    the experiment is run.

    Parameters
    ----------
    project : Project
        The :class:`~skore.Project` that stores the row.
    stem : str
        Unique report key. A duplicate stem is rejected.
    question : str
        The question the experiment answers.
    status : {"aside", "backlog"}, default="aside"
        The status of the new row.
    plan : str or None, default=None
        How to run the experiment, or ``None``.
    source_id : str or None, default=None
        Report or experiment id that raised this row, or ``None``.
    check_code : str or None, default=None
        Check code that raised this row, or ``None``. The same source and
        check together are rejected.

    Returns
    -------
    experiment : Experiment
        The new row. Its ``id`` is the UUID7 the report will carry.
    """
    if status not in {"aside", "backlog"}:
        raise ValueError(f"status must be 'aside' or 'backlog' (found {status!r}).")
    now = datetime.now(UTC).isoformat()
    return project._create_experiment(
        _normalize_row(
            {
                "id": str(uuid7()),
                "stem": stem,
                "status": status,
                "question": question,
                "plan": plan,
                "source_id": source_id,
                "check_code": check_code,
                "abandon_reason": None,
                "version": 1,
                "created_at": now,
                "updated_at": now,
            }
        )
    )
