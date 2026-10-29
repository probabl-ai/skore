"""Render the project journal from the objective and the experiments frame."""

from __future__ import annotations

from pandas import DataFrame, isna

from skore._project.experiments import EXPERIMENT_COLUMNS

_METADATA = frozenset(
    {
        "key",
        "date",
        "learner",
        "dataset",
        "ml_task",
        "report_type",
        "report_id",
        "id",
    }
)
_IDEA_STATUS = {"idea": "open", "aside": "aside", "discarded": "discarded"}
_HISTORY_STATUS = {"history": "done"}


def _cell(value: object) -> str:
    if value is None or (isinstance(value, float) and isna(value)) or isna(value):
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _source(row: dict[str, object], experiments: DataFrame) -> str:
    source_id = _cell(row.get("source_id"))
    check_code = _cell(row.get("check_code"))
    if not source_id and not check_code:
        return "user"
    parent = experiments.loc[experiments["id"].astype(str) == source_id]
    stem = _cell(parent.iloc[0]["stem"]) if not parent.empty else source_id
    if check_code:
        return f"audit:{stem}:checks.{check_code}"
    return f"design:{stem}"


def _table(headers: list[str], rows: list[list[object]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def _headline(summary: DataFrame, experiment_id: str) -> str:
    if summary.empty or "id" not in getattr(summary.index, "names", []):
        return ""
    try:
        matched = summary.xs(experiment_id, level="id")
    except KeyError:
        return ""
    if isinstance(matched, DataFrame):
        if matched.empty:
            return ""
        record = matched.iloc[0]
    else:
        record = matched
    for name in ("roc_auc", "rmse", "log_loss"):
        if name in record.index and not isna(record[name]):
            return f"{name}={record[name]}"
    for name in record.index:
        if name in _METADATA or str(name).endswith("_std"):
            continue
        value = record[name]
        if not isna(value):
            return f"{name}={value}"
    return ""


def render_journal(
    objective: str,
    experiments: DataFrame,
    summary: DataFrame | None = None,
    *,
    report_markdown: dict[str, str] | None = None,
) -> str:
    """Render objective, ideas, backlog, and the selected history rows."""
    frame = experiments.copy()
    if frame.empty:
        frame = DataFrame(columns=list(EXPERIMENT_COLUMNS))
    lines = ["# Objective", "", objective.strip(), ""]

    ideas = frame[frame["status"].isin(_IDEA_STATUS)]
    lines.append("# Ideas")
    lines.append("")
    lines.append(
        _table(
            ["Question", "Status", "Experiment", "Source"],
            [
                [
                    row["question"],
                    _IDEA_STATUS[str(row["status"])],
                    row["stem"],
                    _source(row.to_dict(), frame),
                ]
                for _, row in ideas.iterrows()
            ],
        )
    )
    lines.append("")

    backlog = frame[frame["status"].eq("backlog")]
    lines.append("# Backlog")
    lines.append("")
    lines.append(
        _table(
            ["#", "Item", "Source"],
            [
                [
                    f"B{number}",
                    row["question"],
                    _source(row.to_dict(), frame),
                ]
                for number, (_, row) in enumerate(backlog.iterrows(), start=1)
            ],
        )
    )
    lines.append("")

    history = frame[frame["status"].eq("history")]
    selected: set[str] | None = None
    if summary is not None and not summary.empty:
        selected = set(summary.index.get_level_values("id").astype(str))
        history = history[history["id"].astype(str).isin(selected)]
    lines.append("# History")
    lines.append("")
    lines.append(
        _table(
            ["Stem", "Intent", "Status", "Headline result", "Report", "Design note"],
            [
                [
                    row["stem"],
                    row["question"],
                    _HISTORY_STATUS["history"],
                    "" if summary is None else _headline(summary, str(row["id"])),
                    row["id"],
                    row["plan"],
                ]
                for _, row in history.iterrows()
            ],
        )
    )

    if report_markdown:
        for experiment_id, markdown in report_markdown.items():
            if selected is not None and experiment_id not in selected:
                continue
            lines.extend(["", markdown])
    return "\n".join(lines).rstrip() + "\n"
