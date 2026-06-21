"""Tests for cross-mode project synchronization."""

from __future__ import annotations

from pathlib import Path

import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from skore import EstimatorReport, Project, SyncResult
from skore._project._sync_result import SyncConflictError


@pytest.fixture
def regression_report() -> EstimatorReport:
    X, y = make_regression(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@pytest.fixture
def alt_regression_report() -> EstimatorReport:
    X, y = make_regression(random_state=7)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7
    )
    return EstimatorReport(
        Ridge(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@pytest.fixture
def workspaces(tmp_path: Path) -> tuple[Path, Path]:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    return left, right


def test_sync_with_put(workspaces, regression_report):
    left_ws, right_ws = workspaces
    source = Project("source", mode="local", workspace=left_ws)
    destination = Project("destination", mode="local", workspace=right_ws)

    source.put("baseline", regression_report)

    result = source.sync_with(destination, direction="put")

    assert isinstance(result, SyncResult)
    assert len(result.put) == 1
    assert result.put[0].key == "baseline"
    assert result.put[0].direction == "put"
    assert len(destination.summarize().frame()) == 1
    assert result.summary() == "put 1, got 0, skipped 0, conflicts 0"


def test_sync_with_get(workspaces, regression_report):
    left_ws, right_ws = workspaces
    source = Project("source", mode="local", workspace=left_ws)
    destination = Project("destination", mode="local", workspace=right_ws)

    source.put("baseline", regression_report)
    source.sync_with(destination, direction="put")

    empty = Project("empty", mode="local", workspace=Path(left_ws / "other"))
    result = empty.sync_with(destination, direction="get")

    assert len(result.got) == 1
    assert len(empty.summarize().frame()) == 1


def test_sync_with_both_is_symmetric(workspaces, regression_report):
    left_ws, right_ws = workspaces
    left = Project("left", mode="local", workspace=left_ws)
    right = Project("right", mode="local", workspace=right_ws)

    left.put("only-left", regression_report)

    result_lr = left.sync_with(right, direction="both")
    result_rl = right.sync_with(left, direction="both")

    assert len(result_lr.put) == 1
    assert len(result_rl.skipped) >= 1
    assert len(left.summarize().frame()) == 1
    assert len(right.summarize().frame()) == 1


def test_sync_with_skips_already_synced(workspaces, regression_report):
    left_ws, right_ws = workspaces
    source = Project("source", mode="local", workspace=left_ws)
    destination = Project("destination", mode="local", workspace=right_ws)

    source.put("baseline", regression_report)
    source.sync_with(destination, direction="put")
    result = source.sync_with(destination, direction="both")

    assert len(result.put) == 0
    assert any(skip.reason == "already_synced" for skip in result.skipped)


def test_sync_with_dry_run(workspaces, regression_report):
    left_ws, right_ws = workspaces
    source = Project("source", mode="local", workspace=left_ws)
    destination = Project("destination", mode="local", workspace=right_ws)

    source.put("baseline", regression_report)

    result = source.sync_with(destination, direction="put", dry_run=True)

    assert len(result.put) == 1
    assert destination.summarize().frame().empty


def test_sync_with_conflict_latest_wins(
    workspaces, regression_report, alt_regression_report
):
    left_ws, right_ws = workspaces
    left = Project("left", mode="local", workspace=left_ws)
    right = Project("right", mode="local", workspace=right_ws)

    left.put("model", regression_report)
    right.put("model", alt_regression_report)

    result = left.sync_with(right, direction="both", on_conflict="latest_wins")

    assert len(result.conflicts) == 1
    assert result.conflicts[0].resolution in {
        "latest_wins_put",
        "latest_wins_get",
    }


def test_sync_with_conflict_skip(workspaces, regression_report, alt_regression_report):
    left_ws, right_ws = workspaces
    left = Project("left", mode="local", workspace=left_ws)
    right = Project("right", mode="local", workspace=right_ws)

    left.put("model", regression_report)
    right.put("model", alt_regression_report)

    result = left.sync_with(right, direction="both", on_conflict="skip")

    assert len(result.conflicts) == 1
    assert result.conflicts[0].resolution is None
    assert len(result.put) == 0
    assert len(result.got) == 0


def test_sync_with_conflict_error(workspaces, regression_report, alt_regression_report):
    left_ws, right_ws = workspaces
    left = Project("left", mode="local", workspace=left_ws)
    right = Project("right", mode="local", workspace=right_ws)

    left.put("model", regression_report)
    right.put("model", alt_regression_report)

    with pytest.raises(SyncConflictError, match="Conflict on key 'model'"):
        left.sync_with(right, direction="both", on_conflict="error")


def test_sync_with_conflict_source_wins(
    workspaces, regression_report, alt_regression_report
):
    left_ws, right_ws = workspaces
    left = Project("left", mode="local", workspace=left_ws)
    right = Project("right", mode="local", workspace=right_ws)

    left.put("model", regression_report)
    right.put("model", alt_regression_report)

    result = left.sync_with(right, direction="both", on_conflict="source_wins")

    assert len(result.put) == 1
    assert result.conflicts[0].resolution == "source_wins"
    synced = right.get(result.put[0].destination_id)
    assert synced.estimator_name_ == regression_report.estimator_name_


def test_sync_with_conflict_keep_both(
    workspaces, regression_report, alt_regression_report
):
    left_ws, right_ws = workspaces
    left = Project("left", mode="local", workspace=left_ws)
    right = Project("right", mode="local", workspace=right_ws)

    left.put("model", regression_report)
    right.put("model", alt_regression_report)

    result = left.sync_with(right, direction="both", on_conflict="keep_both")

    assert {entry.key for entry in result.got} == {"model (from right)"}
    assert {entry.key for entry in result.put} == {"model (from left)"}
    assert result.conflicts[0].resolution == "keep_both"
    assert {row["key"] for _, row in left.summarize().frame().iterrows()} == {
        "model",
        "model (from right)",
    }
    assert {row["key"] for _, row in right.summarize().frame().iterrows()} == {
        "model",
        "model (from left)",
    }


def test_sync_with_conflict_keep_both_get_only(
    workspaces, regression_report, alt_regression_report
):
    left_ws, right_ws = workspaces
    left = Project("left", mode="local", workspace=left_ws)
    right = Project("right", mode="local", workspace=right_ws)

    left.put("model", regression_report)
    right.put("model", alt_regression_report)

    result = left.sync_with(right, direction="get", on_conflict="keep_both")

    assert {entry.key for entry in result.got} == {"model (from right)"}
    assert len(result.put) == 0


def test_sync_with_conflict_filtered_by_direction(
    workspaces, regression_report, alt_regression_report
):
    left_ws, right_ws = workspaces
    left = Project("left", mode="local", workspace=left_ws)
    right = Project("right", mode="local", workspace=right_ws)

    left.put("model", regression_report)
    right.put("model", alt_regression_report)

    # source_wins wants to put, but direction only allows getting.
    result = left.sync_with(right, direction="get", on_conflict="source_wins")

    assert len(result.put) == 0
    assert len(result.got) == 0
    assert len(result.conflicts) == 1
    assert result.conflicts[0].resolution is None


def test_sync_summary_counts_only_unresolved_conflicts(
    workspaces, regression_report, alt_regression_report
):
    left_ws, right_ws = workspaces
    left = Project("left", mode="local", workspace=left_ws)
    right = Project("right", mode="local", workspace=right_ws)

    left.put("model", regression_report)
    right.put("model", alt_regression_report)

    result = left.sync_with(right, direction="both", on_conflict="latest_wins")

    # The conflict is auto-resolved into a transfer, so it must not inflate the
    # conflict count reported by ``summary()``.
    assert len(result.conflicts) == 1
    assert "conflicts 0" in result.summary()
    assert (len(result.put) + len(result.got)) == 1


def test_sync_with_resync_does_not_duplicate_after_registry_loss(
    workspaces, regression_report
):
    import shutil

    left_ws, right_ws = workspaces
    source = Project("source", mode="local", workspace=left_ws)
    destination = Project("destination", mode="local", workspace=right_ws)

    source.put("baseline", regression_report)
    source.sync_with(destination, direction="put")

    # Drop the registry to simulate a fresh machine; the canonical report id must
    # still prevent a duplicate transfer.
    shutil.rmtree(left_ws / "sync", ignore_errors=True)
    shutil.rmtree(right_ws / "sync", ignore_errors=True)

    result = source.sync_with(destination, direction="both")

    assert len(result.put) == 0
    assert len(result.got) == 0
    assert len(destination.summarize().frame()) == 1


def test_sync_with_cannot_sync_with_self(workspaces):
    left_ws, _ = workspaces
    project = Project("solo", mode="local", workspace=left_ws)

    with pytest.raises(ValueError, match="Cannot synchronize a project with itself"):
        project.sync_with(project)


def test_sync_with_ml_task_mismatch(workspaces, regression_report):
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    left_ws, right_ws = workspaces
    left = Project("left", mode="local", workspace=left_ws)
    right = Project("right", mode="local", workspace=right_ws)

    left.put("regression", regression_report)

    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    classification = EstimatorReport(
        LogisticRegression(max_iter=10_000),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    right.put("classification", classification)

    with pytest.raises(ValueError, match="different ML tasks"):
        left.sync_with(right)


def test_sync_with_mode_string(workspaces, regression_report):
    left_ws, right_ws = workspaces
    source = Project("shared", mode="local", workspace=left_ws)
    source.put("baseline", regression_report)

    # The counterpart is built as Project("shared", mode="local", workspace=right_ws).
    result = source.sync_with("local", workspace=right_ws, direction="put")

    assert len(result.put) == 1
    destination = Project("shared", mode="local", workspace=right_ws)
    assert len(destination.summarize().frame()) == 1


def test_sync_with_kwargs_requires_mode_string(workspaces, regression_report):
    left_ws, right_ws = workspaces
    source = Project("source", mode="local", workspace=left_ws)
    destination = Project("destination", mode="local", workspace=right_ws)

    with pytest.raises(TypeError, match="only supported when `other` is a mode string"):
        source.sync_with(destination, workspace=right_ws)


def test_sync_result_summary():
    result = SyncResult(put=(), got=(), skipped=(), conflicts=(), failed=())
    assert result.summary() == "put 0, got 0, skipped 0, conflicts 0"
