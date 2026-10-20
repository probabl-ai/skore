"""Create experiments. This is the only public way to add one."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skore._project.experiments import _blank, normalize_row
from skore._utils.uuid import uuid7

if TYPE_CHECKING:
    from skore._project.experiments import Experiment
    from skore._project.project import Project


def create_experiment(
    project: Project,
    stem: str,
    question: str,
    plan: str | None = None,
    source_id: str | None = None,
    check_code: str | None = None,
) -> Experiment:
    """Append one experiment and return a handle on it.

    Passing ``plan`` creates the row directly in the backlog. Omitting it
    creates an idea. The returned id is the UUID7 the report will carry
    when the experiment is run.
    """
    text_plan = _blank(plan)
    frame = project.experiments.frame
    cleaned_stem = stem.strip()
    if not frame.empty and frame["stem"].astype(str).eq(cleaned_stem).any():
        raise ValueError(f"An experiment with stem {cleaned_stem!r} already exists.")
    cleaned_check = _blank(check_code)
    cleaned_source = _blank(source_id)
    if cleaned_check is not None and not frame.empty:
        same_check = frame["check_code"].fillna("").astype(str).eq(cleaned_check)
        same_source = frame["source_id"].fillna("").astype(str).eq(cleaned_source or "")
        if (same_check & same_source).any():
            raise ValueError(
                "An experiment already exists for "
                f"source {cleaned_source!r} and check {cleaned_check!r}."
            )
    row = normalize_row(
        {
            "id": str(uuid7()),
            "stem": cleaned_stem,
            "status": "backlog" if text_plan is not None else "idea",
            "question": question,
            "source_id": cleaned_source,
            "check_code": cleaned_check,
            "plan": text_plan,
        }
    )
    return project._write_experiment(row)
