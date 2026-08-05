from itertools import permutations
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import mlflow
import pytest
from pandas import NA, DataFrame, Index, MultiIndex, RangeIndex, isna
from pandas.testing import assert_frame_equal
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge

from skore import EstimatorReport, Project, evaluate
from skore._project._sync import synchronize


class FakeSummary:
    def __init__(self, frame: DataFrame):
        self._frame = frame

    def frame(self) -> DataFrame:
        return self._frame


class FakeProject:
    def __init__(self, mode, records=(), reports=None):
        self.mode = mode
        self.name = "project"
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

    def summarize(self) -> FakeSummary:
        frame = DataFrame(sorted(self.records, key=lambda row: row["date"]))
        if not frame.empty:
            frame.index = MultiIndex.from_arrays(
                [
                    RangeIndex(len(frame)),
                    Index(frame.pop("id"), name="id", dtype=str),
                ]
            )
        return FakeSummary(frame)


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

    expected = DataFrame(
        {
            "key": ["model"],
            "direction": ["outbound"],
            "status": ["transferred"],
        },
        index=Index(["report-1"], name="report_id"),
        dtype="string",
    )
    assert_frame_equal(result, expected)
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

    expected = DataFrame(
        {
            "key": ["left-only", "right-only", "shared"],
            "direction": ["outbound", "inbound", NA],
            "status": ["transferred", "transferred", "skipped"],
        },
        index=Index(["report-1", "report-3", "report-2"], name="report_id"),
        dtype="string",
    )
    assert_frame_equal(result, expected)
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

    assert result.index.tolist() == ["report-1"]
    assert result.loc["report-1", "key"] == "left-key"
    assert isna(result.loc["report-1", "direction"])
    assert result.loc["report-1", "status"] == "skipped"
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

    assert result.loc["report-1", "key"] == "latest-key"
    source.get.assert_called_once_with("latest")
    destination.put.assert_called_once_with("latest-key", latest)


def test_synchronize_ignores_missing_report_ids():
    identified_report = report("report-1")
    source = FakeProject(
        "local",
        [
            record("unidentified", None, "legacy", "2026-01-01"),
            record("identified", "report-1", "model", "2026-01-02"),
        ],
        {
            "unidentified": report("legacy-report"),
            "identified": identified_report,
        },
    )
    destination = FakeProject("hub")

    result = synchronize(source, destination, bidirectional=False, dry_run=False)

    assert result.index.tolist() == ["report-1"]
    source.get.assert_called_once_with("identified")
    destination.put.assert_called_once_with("model", identified_report)


def test_synchronize_ignores_missing_report_id_column():
    source = FakeProject(
        "local",
        [{"id": "legacy", "key": "legacy", "date": "2026-01-01"}],
        {"legacy": report("legacy-report")},
    )
    destination = FakeProject("hub")

    result = synchronize(source, destination, bidirectional=False, dry_run=False)

    expected = DataFrame(
        index=Index([], name="report_id", dtype=object),
        columns=Index(["key", "direction", "status"]),
    ).astype("string")
    assert_frame_equal(result, expected)
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

    assert result.index.tolist() == ["report-1"]
    assert result.loc["report-1", "direction"] == "outbound"
    assert result.loc["report-1", "status"] == "planned"
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

    assert result.index.tolist() == ["report-2", "report-1"]
    assert result.loc["report-2", "status"] == "transferred"
    assert result.loc["report-1", "status"] == "skipped"
    source.get.assert_called_once_with("source-2")


@pytest.fixture
def mlflow_tracking_uri(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    previous_tracking_uri = mlflow.get_tracking_uri()
    tracking_path = tmp_path / "mlruns"
    tracking_path.mkdir()
    tracking_uri = tracking_path.as_uri()
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
@pytest.mark.filterwarnings(
    "ignore:The filesystem tracking backend .* is deprecated:FutureWarning"
)
def test_sync_local_and_mlflow_bidirectionally(tmp_path, mlflow_tracking_uri):
    regression_data = make_regression(random_state=42, coef=False)
    X, y = regression_data[0], regression_data[1]
    local_report = cast(EstimatorReport, evaluate(LinearRegression(), X, y))
    mlflow_report = cast(EstimatorReport, evaluate(Ridge(), X, y))
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

    assert set(dry_run["direction"]) == {"outbound", "inbound"}
    assert set(dry_run["status"]) == {"planned"}
    assert len(local.summarize().frame()) == 1
    assert len(mlflow_project.summarize().frame()) == 1

    result = local.sync(
        mlflow_project,
        bidirectional=True,
    )

    assert set(result["status"]) == {"transferred"}
    expected_ids = {str(local_report.id), str(mlflow_report.id)}
    assert set(local.summarize().frame()["report_id"]) == expected_ids
    assert set(mlflow_project.summarize().frame()["report_id"]) == expected_ids

    repeated = local.sync(
        mlflow_project,
        bidirectional=True,
    )

    assert set(repeated.index) == expected_ids
    assert bool(repeated["direction"].isna().all())
    assert set(repeated["status"]) == {"skipped"}
