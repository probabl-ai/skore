from itertools import permutations
from types import SimpleNamespace
from unittest.mock import Mock

import mlflow
import pytest
from pandas import DataFrame, Index, MultiIndex, RangeIndex
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge

from skore import Project, SyncOperation, evaluate
from skore._project._sync import synchronize


class FakeProject:
    def __init__(self, mode, records=(), reports=None, ml_task="regression"):
        self.mode = mode
        self.name = "project"
        self.ml_task = ml_task
        self.records = list(records)
        self.reports = {} if reports is None else dict(reports)
        self.get = Mock(side_effect=self._get)
        self.put = Mock(side_effect=self._put)

    def _get(self, backend_id):
        return self.reports[backend_id]

    def _put(self, key, report):
        backend_id = f"{self.mode}-{len(self.records)}"
        self.records.append(
            {
                "id": backend_id,
                "report_id": str(report.id),
                "key": key,
                "date": f"2026-01-{len(self.records) + 1:02d}",
            }
        )
        self.reports[backend_id] = report

    def summarize(self):
        frame = DataFrame(sorted(self.records, key=lambda row: row["date"]))
        if not frame.empty:
            frame.index = MultiIndex.from_arrays(
                [
                    RangeIndex(len(frame)),
                    Index(frame.pop("id"), name="id", dtype=str),
                ]
            )
        return SimpleNamespace(frame=lambda: frame)


def record(backend_id, report_id, key, date="2026-01-01"):
    return {
        "id": backend_id,
        "report_id": report_id,
        "key": key,
        "date": date,
    }


def report(report_id):
    return SimpleNamespace(id=report_id)


def report_ids(project):
    return {row["report_id"] for row in project.records}


@pytest.mark.parametrize(
    ("source_mode", "destination_mode"), permutations(("local", "hub", "mlflow"), 2)
)
def test_synchronize_supports_all_mode_pairs(source_mode, destination_mode):
    source_report = report("report-1")
    source = FakeProject(
        source_mode,
        [record("source-1", "report-1", "model")],
        {"source-1": source_report},
    )
    destination = FakeProject(destination_mode)

    result = synchronize(source, destination, bidirectional=False, dry_run=False)

    assert result.operations == (
        SyncOperation(
            report_id="report-1",
            key="model",
            source_mode=source_mode,
            destination_mode=destination_mode,
        ),
    )
    assert result.skipped == ()
    assert result.dry_run is False
    source.get.assert_called_once_with("source-1")
    destination.put.assert_called_once_with("model", source_report)


def test_synchronize_bidirectionally_uses_initial_snapshots():
    left = FakeProject(
        "local",
        [
            record("left-1", "report-1", "left-only", "2026-01-01"),
            record("left-2", "report-2", "shared", "2026-01-02"),
        ],
        {
            "left-1": report("report-1"),
            "left-2": report("report-2"),
        },
    )
    right = FakeProject(
        "hub",
        [
            record("right-2", "report-2", "another-key", "2026-01-01"),
            record("right-3", "report-3", "right-only", "2026-01-02"),
        ],
        {
            "right-2": report("report-2"),
            "right-3": report("report-3"),
        },
    )

    result = synchronize(left, right, bidirectional=True, dry_run=False)

    assert [
        (operation.report_id, operation.source_mode, operation.destination_mode)
        for operation in result.operations
    ] == [
        ("report-1", "local", "hub"),
        ("report-3", "hub", "local"),
    ]
    assert result.skipped == ("report-2",)
    assert report_ids(left) == {"report-1", "report-2", "report-3"}
    assert report_ids(right) == {"report-1", "report-2", "report-3"}


def test_synchronize_skips_same_id_without_loading_reports():
    left = FakeProject(
        "local",
        [record("left", "report-1", "left-key")],
        {"left": report("report-1")},
    )
    right = FakeProject(
        "hub",
        [record("right", "report-1", "right-key")],
        {"right": report("different-state")},
    )

    result = synchronize(left, right, bidirectional=True, dry_run=False)

    assert result.operations == ()
    assert result.skipped == ("report-1",)
    left.get.assert_not_called()
    right.get.assert_not_called()


def test_synchronize_allows_different_ids_with_same_key():
    left_report = report("report-1")
    left = FakeProject(
        "local",
        [record("left", "report-1", "model")],
        {"left": left_report},
    )
    right = FakeProject(
        "hub",
        [record("right", "report-2", "model")],
        {"right": report("report-2")},
    )

    synchronize(left, right, bidirectional=False, dry_run=False)

    right.put.assert_called_once_with("model", left_report)


def test_synchronize_uses_latest_source_duplicate():
    latest = report("report-1")
    source = FakeProject(
        "local",
        [
            record("old", "report-1", "old-key", "2026-01-01"),
            record("latest", "report-1", "latest-key", "2026-01-02"),
        ],
        {"old": report("report-1"), "latest": latest},
    )
    destination = FakeProject("hub")

    result = synchronize(source, destination, bidirectional=False, dry_run=False)

    assert result.operations[0].key == "latest-key"
    source.get.assert_called_once_with("latest")
    destination.put.assert_called_once_with("latest-key", latest)


@pytest.mark.parametrize("missing_side", ["source", "destination"])
def test_synchronize_rejects_missing_canonical_id_before_transfer(missing_side):
    identified = FakeProject(
        "local",
        [record("identified", "report-1", "model")],
        {"identified": report("report-1")},
    )
    unidentified = FakeProject(
        "hub",
        [record("unidentified", None, "legacy")],
        {"unidentified": report("legacy-report")},
    )
    source, destination = (
        (unidentified, identified)
        if missing_side == "source"
        else (identified, unidentified)
    )

    with pytest.raises(ValueError, match="no canonical `report_id`"):
        synchronize(source, destination, bidirectional=False, dry_run=False)

    source.get.assert_not_called()
    destination.put.assert_not_called()


def test_synchronize_dry_run_does_not_load_or_store_reports():
    source = FakeProject(
        "local",
        [record("source", "report-1", "model")],
        {"source": report("report-1")},
    )
    destination = FakeProject("mlflow")

    result = synchronize(source, destination, bidirectional=False, dry_run=True)

    assert result.dry_run is True
    assert [operation.report_id for operation in result.operations] == ["report-1"]
    source.get.assert_not_called()
    destination.put.assert_not_called()


def test_synchronize_rejects_loaded_report_id_mismatch():
    source = FakeProject(
        "local",
        [record("source", "expected", "model")],
        {"source": report("actual")},
    )
    destination = FakeProject("hub")

    with pytest.raises(RuntimeError, match="does not match its project summary"):
        synchronize(source, destination, bidirectional=False, dry_run=False)

    destination.put.assert_not_called()


def test_synchronize_rejects_different_ml_tasks_before_transfer():
    source = FakeProject(
        "local",
        [record("source", "report-1", "model")],
        {"source": report("report-1")},
        ml_task="regression",
    )
    destination = FakeProject("hub", ml_task="binary-classification")

    with pytest.raises(ValueError, match="different ML tasks"):
        synchronize(source, destination, bidirectional=False, dry_run=False)

    source.get.assert_not_called()
    destination.put.assert_not_called()


def test_synchronize_stops_on_first_transfer_error():
    source = FakeProject(
        "local",
        [
            record("source-1", "report-1", "first", "2026-01-01"),
            record("source-2", "report-2", "second", "2026-01-02"),
        ],
        {
            "source-1": report("report-1"),
            "source-2": report("report-2"),
        },
    )
    destination = FakeProject("hub")
    put_calls = 0

    def fail_second_put(key, transferred_report):
        nonlocal put_calls
        put_calls += 1
        if put_calls == 2:
            raise RuntimeError("upload failed")
        destination._put(key, transferred_report)

    destination.put.side_effect = fail_second_put

    with pytest.raises(RuntimeError, match="upload failed"):
        synchronize(source, destination, bidirectional=False, dry_run=False)

    assert source.get.call_count == 2
    assert destination.put.call_count == 2
    assert report_ids(destination) == {"report-1"}

    source.get.reset_mock()
    destination.put.reset_mock(side_effect=True)
    destination.put.side_effect = destination._put

    result = synchronize(source, destination, bidirectional=False, dry_run=False)

    assert [operation.report_id for operation in result.operations] == ["report-2"]
    source.get.assert_called_once_with("source-2")


@pytest.fixture
def mlflow_tracking_uri(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    previous_tracking_uri = mlflow.get_tracking_uri()
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    try:
        yield tracking_uri
    finally:
        while mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous_tracking_uri)


@pytest.mark.filterwarnings(
    r"ignore:codecs\.open\(\) is deprecated:DeprecationWarning:mlflow"
)
def test_sync_local_and_mlflow_bidirectionally(tmp_path, mlflow_tracking_uri):
    X, y = make_regression(random_state=42)
    local_report = evaluate(LinearRegression(), X, y)
    mlflow_report = evaluate(Ridge(), X, y)
    local = Project(
        name="sync-project",
        mode="local",
        workspace=tmp_path / "local",
    )
    mlflow_project = Project(
        name="sync-project",
        mode="mlflow",
        tracking_uri=mlflow_tracking_uri,
    )
    local.put("local-model", local_report)
    mlflow_project.put("mlflow-model", mlflow_report)

    dry_run = local.sync(
        mlflow_project,
        bidirectional=True,
        dry_run=True,
    )

    assert {
        (operation.source_mode, operation.destination_mode)
        for operation in dry_run.operations
    } == {("local", "mlflow"), ("mlflow", "local")}
    assert len(local.summarize().frame()) == 1
    assert len(mlflow_project.summarize().frame()) == 1

    result = local.sync(
        mlflow_project,
        bidirectional=True,
    )

    assert result.dry_run is False
    expected_ids = {str(local_report.id), str(mlflow_report.id)}
    assert set(local.summarize().frame()["report_id"]) == expected_ids
    assert set(mlflow_project.summarize().frame()["report_id"]) == expected_ids

    repeated = local.sync(
        mlflow_project,
        bidirectional=True,
    )

    assert repeated.operations == ()
    assert set(repeated.skipped) == expected_ids
