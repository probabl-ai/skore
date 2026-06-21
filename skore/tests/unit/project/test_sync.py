"""Tests for cross-mode project synchronization."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pandas import NaT, Timestamp
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from skore import EstimatorReport, Project, SyncResult
from skore._plugins.local.sync_registry import SyncRegistry
from skore._project._sync_result import SyncConflictError
from skore._project.sync import (
    _Op,
    _register_pair_link,
    _registry_linked,
    _ReportRef,
    _resolve_conflict,
    _resolve_destination_id,
    _row_id,
    _run_ops,
    _sync_registry_for_pair,
)


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


def test_sync_with_same_backend_is_rejected(workspaces):
    left_ws, _ = workspaces
    # Two distinct objects pointing at the same name + workspace are the same backend.
    a = Project("dup", mode="local", workspace=left_ws)
    b = Project("dup", mode="local", workspace=left_ws)

    with pytest.raises(ValueError, match="Cannot synchronize a project with itself"):
        a.sync_with(b)


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


def test_sync_result_summary_reports_failures():
    result = SyncResult(
        put=(),
        got=(),
        skipped=(),
        conflicts=(),
        failed=(("model", RuntimeError("boom")),),
    )

    assert "failed 1" in result.summary()


# ---------------------------------------------------------------------------
# Direct unit tests for the pure helpers in ``skore._project.sync``.
# ---------------------------------------------------------------------------


def _ref(report_id: str, date: Timestamp) -> _ReportRef:
    return _ReportRef(key="model", id=report_id, date=date)


def test_row_id_handles_flat_and_multiindex():
    assert _row_id(("model", "the-id")) == "the-id"
    assert _row_id("the-id") == "the-id"


def test_resolve_conflict_destination_wins():
    action, resolution = _resolve_conflict(
        on_conflict="destination_wins",
        self_ref=_ref("a", Timestamp("2024-01-01", tz="UTC")),
        other_ref=_ref("b", Timestamp("2024-01-02", tz="UTC")),
    )

    assert action == "get"
    assert resolution == "destination_wins"


@pytest.mark.parametrize(
    ("self_date", "other_date", "expected_action", "expected_resolution"),
    [
        (NaT, NaT, "put", "latest_wins_put"),
        (NaT, Timestamp("2024-01-01", tz="UTC"), "get", "latest_wins_get"),
        (Timestamp("2024-01-01", tz="UTC"), NaT, "put", "latest_wins_put"),
        (
            Timestamp("2024-01-02", tz="UTC"),
            Timestamp("2024-01-01", tz="UTC"),
            "put",
            "latest_wins_put",
        ),
        (
            Timestamp("2024-01-01", tz="UTC"),
            Timestamp("2024-01-02", tz="UTC"),
            "get",
            "latest_wins_get",
        ),
    ],
)
def test_resolve_conflict_latest_wins(
    self_date, other_date, expected_action, expected_resolution
):
    action, resolution = _resolve_conflict(
        on_conflict="latest_wins",
        self_ref=_ref("a", self_date),
        other_ref=_ref("b", other_date),
    )

    assert action == expected_action
    assert resolution == expected_resolution


def test_resolve_conflict_unknown_policy_raises():
    with pytest.raises(ValueError, match="Unknown conflict policy"):
        _resolve_conflict(
            on_conflict="bogus",  # type: ignore[arg-type]
            self_ref=_ref("a", Timestamp("2024-01-01", tz="UTC")),
            other_ref=_ref("b", Timestamp("2024-01-02", tz="UTC")),
        )


def test_resolve_destination_id_fallbacks(workspaces, regression_report):
    left_ws, _ = workspaces
    project = Project("dest", mode="local", workspace=left_ws)
    project.put("k", regression_report)

    report_id = _row_id(project.summarize().frame().index[0])

    # The freshly written id is detected against an empty baseline.
    assert _resolve_destination_id(project, "k", set()) == report_id
    # Nothing new for the key: fall back to the latest existing id for that key.
    assert _resolve_destination_id(project, "k", {report_id}) == report_id
    # The key matches no row: fall back to a globally new id.
    assert _resolve_destination_id(project, "missing", set()) == report_id


def test_resolve_destination_id_raises_when_not_found(workspaces, regression_report):
    left_ws, _ = workspaces
    project = Project("dest", mode="local", workspace=left_ws)
    project.put("k", regression_report)

    report_id = _row_id(project.summarize().frame().index[0])

    with pytest.raises(RuntimeError, match="Could not locate the report"):
        _resolve_destination_id(project, "missing", {report_id})


def test_run_ops_captures_per_report_failures():
    class _BoomOrigin:
        mode = "local"

        def get(self, _report_id):
            raise RuntimeError("boom")

    op = _Op(
        direction="put",
        origin=_BoomOrigin(),
        destination=SimpleNamespace(mode="local"),
        origin_uri="origin",
        destination_uri="destination",
        ref=_ref("rid", Timestamp("2024-01-01", tz="UTC")),
        key="model",
    )

    put, got, failed = _run_ops([op], None, dry_run=False)

    assert put == []
    assert got == []
    assert len(failed) == 1
    assert failed[0][0] == "model"
    assert isinstance(failed[0][1], RuntimeError)


def test_registry_linked_without_registry_returns_false():
    ref = _ref("i", Timestamp("2024-01-01", tz="UTC"))

    assert (
        _registry_linked(
            None,
            self_proj=SimpleNamespace(mode="local"),
            self_ref=ref,
            other_proj=SimpleNamespace(mode="local"),
            other_ref=ref,
            self_uri="a",
            other_uri="b",
        )
        is False
    )


def test_registry_linked_two_remote_projects_returns_false(tmp_path):
    registry = SyncRegistry(tmp_path)
    ref = _ref("i", Timestamp("2024-01-01", tz="UTC"))

    assert (
        _registry_linked(
            registry,
            self_proj=SimpleNamespace(mode="hub"),
            self_ref=ref,
            other_proj=SimpleNamespace(mode="mlflow"),
            other_ref=ref,
            self_uri="a",
            other_uri="b",
        )
        is False
    )


def test_register_pair_link_without_registry_is_noop():
    ref = _ref("i", Timestamp("2024-01-01", tz="UTC"))

    # Must not raise nor touch any storage.
    _register_pair_link(
        None,
        self_proj=SimpleNamespace(mode="local"),
        self_ref=ref,
        other_proj=SimpleNamespace(mode="local"),
        other_ref=ref,
        self_uri="a",
        other_uri="b",
    )


def test_sync_registry_for_pair_returns_none_for_two_remotes():
    assert (
        _sync_registry_for_pair(
            SimpleNamespace(mode="hub"),
            SimpleNamespace(mode="mlflow"),
        )
        is None
    )


def test_sync_registry_linked_with_existing_record(tmp_path):
    registry = SyncRegistry(tmp_path)
    registry.link(
        report_id="rid",
        key="model",
        remote_uri="remote-a",
        remote_id="remote-rid",
    )

    assert registry.linked(
        report_id="rid",
        key="model",
        remote_uri="remote-a",
        remote_id="remote-rid",
    )
    # Same record, but a different remote_uri has no link.
    assert not registry.linked(
        report_id="rid",
        key="model",
        remote_uri="remote-b",
        remote_id="remote-rid",
    )


# ---------------------------------------------------------------------------
# Engine branches reachable with two local projects.
# ---------------------------------------------------------------------------


def test_sync_with_conflict_destination_wins(
    workspaces, regression_report, alt_regression_report
):
    left_ws, right_ws = workspaces
    left = Project("left", mode="local", workspace=left_ws)
    right = Project("right", mode="local", workspace=right_ws)

    left.put("model", regression_report)
    right.put("model", alt_regression_report)

    result = left.sync_with(right, direction="both", on_conflict="destination_wins")

    assert len(result.put) == 0
    assert len(result.got) == 1
    assert result.conflicts[0].resolution == "destination_wins"


def test_sync_with_keep_both_skips_existing_alt_keys(
    workspaces, regression_report, alt_regression_report
):
    left_ws, right_ws = workspaces
    left = Project("left", mode="local", workspace=left_ws)
    right = Project("right", mode="local", workspace=right_ws)

    # ``model`` diverges, triggering keep_both.
    left.put("model", regression_report)
    right.put("model", alt_regression_report)

    # The alternate keys already exist on both sides with identical reports, so
    # they dedup (no transfer) and the keep_both guards take their False branch.
    left.put("model (from right)", regression_report)
    right.put("model (from right)", regression_report)
    left.put("model (from left)", regression_report)
    right.put("model (from left)", regression_report)

    result = left.sync_with(right, direction="both", on_conflict="keep_both")

    assert len(result.put) == 0
    assert len(result.got) == 0
    keep_both = [c for c in result.conflicts if c.resolution == "keep_both"]
    assert [c.key for c in keep_both] == ["model"]


# ---------------------------------------------------------------------------
# Cross-mode integration tests (local <-> mlflow).
# ---------------------------------------------------------------------------


@pytest.fixture
def mlflow_tracking(tmp_path, monkeypatch):
    import mlflow

    monkeypatch.chdir(tmp_path)
    previous_tracking_uri = mlflow.get_tracking_uri()
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    try:
        yield tracking_uri
    finally:
        while mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous_tracking_uri)


def test_sync_local_to_mlflow_dedups_on_resync(
    workspaces, regression_report, mlflow_tracking
):
    left_ws, _ = workspaces
    local = Project("xp", mode="local", workspace=left_ws)
    ml = Project("xp", mode="mlflow", tracking_uri=mlflow_tracking)

    local.put("baseline", regression_report)
    result = local.sync_with(ml, direction="put")
    assert len(result.put) == 1

    # The mlflow summarize id differs from the canonical report id, so dedup must
    # rely on the persisted registry link rather than id equality.
    resynced = local.sync_with(ml, direction="both")

    assert len(resynced.put) == 0
    assert len(resynced.got) == 0
    assert any(skip.reason == "already_synced" for skip in resynced.skipped)
    assert len(ml.summarize().frame()) == 1


def test_sync_local_mlflow_dedups_after_registry_loss(
    workspaces, regression_report, mlflow_tracking
):
    import shutil

    left_ws, _ = workspaces
    local = Project("xp", mode="local", workspace=left_ws)
    ml = Project("xp", mode="mlflow", tracking_uri=mlflow_tracking)

    local.put("baseline", regression_report)
    local.sync_with(ml, direction="put")

    # Drop the registry: dedup must fall back to the canonical report id fetched
    # from the remote backend.
    shutil.rmtree(left_ws / "sync", ignore_errors=True)

    result = local.sync_with(ml, direction="both")

    assert len(result.put) == 0
    assert len(result.got) == 0
    assert any(skip.reason == "already_synced" for skip in result.skipped)
    assert len(ml.summarize().frame()) == 1


def test_sync_mlflow_to_local_dedups_from_remote_side(
    workspaces, regression_report, mlflow_tracking
):
    import shutil

    left_ws, _ = workspaces
    local = Project("xp", mode="local", workspace=left_ws)
    ml = Project("xp", mode="mlflow", tracking_uri=mlflow_tracking)

    local.put("baseline", regression_report)
    local.sync_with(ml, direction="put")

    # ``self`` is the mlflow project and ``other`` is local: the registry lookup
    # must use the local (other) side.
    resynced = ml.sync_with(local, direction="both")
    assert len(resynced.put) == 0
    assert len(resynced.got) == 0
    assert any(skip.reason == "already_synced" for skip in resynced.skipped)

    # After registry loss the canonical id still dedups from the mlflow side.
    shutil.rmtree(left_ws / "sync", ignore_errors=True)
    after_loss = ml.sync_with(local, direction="both")

    assert len(after_loss.put) == 0
    assert len(after_loss.got) == 0
    assert any(skip.reason == "already_synced" for skip in after_loss.skipped)
    assert len(local.summarize().frame()) == 1
