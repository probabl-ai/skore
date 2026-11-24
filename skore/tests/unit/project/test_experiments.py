"""Lifecycle, conflicts, and local storage of project experiments."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import mkstemp

import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

from skore import Project
from skore._project.experiments import JournalConflictError
from skore.agent import create_experiment

_CHILD = """
import json
import sys

from skore import Project
from skore._project.experiments import JournalConflictError

workspace, name, payload = sys.argv[1:]
project = Project(name=name, mode="local", workspace=workspace)
row = json.loads(payload)
row["status"] = "running"
try:
    project._update_experiment(row, on_same_status="conflict")
except JournalConflictError:
    print("conflict")
else:
    print("ok")
"""


def _project(tmp_path: Path) -> Project:
    return Project(name="experiments", mode="local", workspace=tmp_path)


def _data():
    return make_regression(n_samples=20, n_features=3, random_state=0)


def test_lifecycle(tmp_path):
    project = _project(tmp_path)
    experiment = create_experiment(project, " encoding ", " Does encoding help? ")
    assert experiment.stem == "encoding"
    assert experiment.question == "Does encoding help?"
    assert experiment.status == "aside"
    assert experiment.version == 1
    assert experiment.plan is None

    planned = experiment.update_plan("try it")
    assert planned.plan == "try it"
    assert planned.version == 2

    discarded = planned.decide("discarded")
    restored = discarded.decide("aside")
    assert restored.status == "aside"

    backlog = restored.promote()
    assert backlog.status == "backlog"
    assert backlog.plan == "try it"
    running = backlog.start()
    released = running.release()
    assert released.status == "backlog"

    running = released.start().update_plan("try it twice")
    assert running.status == "running"
    assert running.plan == "try it twice"

    X, y = _data()
    report = running.run(Ridge(random_state=0), X, y)
    finished = project.experiments.get(experiment.id)
    assert finished.status == "history"
    assert report.id == experiment.id
    assert project.get(experiment.id).id == experiment.id
    with pytest.raises(ValueError):
        finished.update_plan("too late")
    with pytest.raises(ValueError):
        finished.run(Ridge(), X, y)


def test_run_from_backlog_and_abandon(tmp_path):
    project = _project(tmp_path)
    experiment = create_experiment(
        project,
        "mean",
        "What does the mean predict?",
        status="backlog",
        plan="fit a ridge model",
        source_id="report-1",
        check_code="LowScore",
    )
    assert experiment.status == "backlog"
    assert experiment.source_id == "report-1"
    assert experiment.check_code == "LowScore"

    X, y = _data()
    report = experiment.run(Ridge(random_state=0), X, y)
    finished = project.experiments.get(experiment.id)
    assert finished.status == "history"
    assert report.id == experiment.id

    other = create_experiment(project, "other", "Another question?", status="backlog")
    abandoned = other.start().abandon(" the split leaked ")
    assert abandoned.status == "abandoned"
    assert abandoned.abandon_reason == "the split leaked"
    with pytest.raises(ValueError):
        abandoned.update_plan("too late")
    with pytest.raises(ValueError):
        abandoned.release()


def test_forbidden_transitions(tmp_path):
    project = _project(tmp_path)
    aside = create_experiment(project, "stem", "question")
    with pytest.raises(ValueError):
        aside.start()
    with pytest.raises(ValueError):
        aside.release()
    with pytest.raises(ValueError):
        aside.abandon("nope")
    with pytest.raises(ValueError):
        aside.decide("backlog")
    X, y = _data()
    with pytest.raises(ValueError):
        aside.run(Ridge(), X, y)

    backlog = aside.promote()
    with pytest.raises(ValueError):
        backlog.decide("discarded")
    running = backlog.start()
    with pytest.raises(ValueError):
        running.promote()
    with pytest.raises(ValueError):
        running.decide("aside")


def test_same_target_decide_and_promote_do_not_write(tmp_path):
    project = _project(tmp_path)
    experiment = create_experiment(project, "stem", "question", plan="first")
    other = project.experiments.get(experiment.id)

    discarded = experiment.decide("discarded")
    again = other.decide("discarded")
    assert again.status == "discarded"
    assert again.version == discarded.version

    aside = again.decide("aside")
    stale = project.experiments.get(experiment.id)
    promoted = aside.promote(plan="kept")
    ignored = stale.promote(plan="ignored")
    assert ignored.status == "backlog"
    assert ignored.plan == "kept"
    assert ignored.version == promoted.version


def test_start_on_a_running_row_conflicts(tmp_path):
    project = _project(tmp_path)
    experiment = create_experiment(project, "stem", "question", status="backlog")
    other = project.experiments.get(experiment.id)
    running = experiment.start()

    with pytest.raises(JournalConflictError) as caught:
        other.start()

    assert caught.value.current["status"] == "running"
    assert caught.value.current["version"] == running.version
    with pytest.raises(JournalConflictError):
        running.start()


def test_stale_plan_and_promote_conflict(tmp_path):
    project = _project(tmp_path)
    experiment = create_experiment(project, "stem", "question")
    other = project.experiments.get(experiment.id)
    planned = experiment.update_plan("one")

    with pytest.raises(JournalConflictError) as caught:
        other.update_plan("two")

    assert caught.value.current["plan"] == "one"
    assert project.experiments.get(experiment.id).plan == "one"
    assert project.experiments.get(experiment.id).version == planned.version

    readable = project.experiments.get(experiment.id)
    planned.decide("discarded")
    with pytest.raises(JournalConflictError) as caught:
        readable.promote()
    assert caught.value.current["status"] == "discarded"


def test_duplicate_stem_and_check_raise(tmp_path):
    project = _project(tmp_path)
    experiment = create_experiment(
        project,
        "stem",
        "question",
        source_id="report-1",
        check_code="LowScore",
    )

    with pytest.raises(JournalConflictError) as caught:
        create_experiment(project, "stem", "another question")
    assert caught.value.current["id"] == experiment.id

    with pytest.raises(JournalConflictError) as caught:
        create_experiment(
            project,
            "other",
            "from the same check",
            source_id="report-1",
            check_code="LowScore",
        )
    assert caught.value.current["id"] == experiment.id

    create_experiment(
        project,
        "different-check",
        "another check",
        source_id="report-1",
        check_code="Other",
    )
    create_experiment(project, "no-check", "no check", source_id="report-1")
    create_experiment(project, "also-no-check", "still no check", source_id="report-1")


def test_blank_fields_and_unknown_status(tmp_path):
    project = _project(tmp_path)
    with pytest.raises(ValueError):
        create_experiment(project, "  ", "question")
    with pytest.raises(ValueError):
        create_experiment(project, "stem", "  ")
    with pytest.raises(ValueError):
        create_experiment(project, "stem", "question", status="running")

    experiment = create_experiment(project, "stem", "question", status="backlog")
    with pytest.raises(ValueError):
        experiment.update_plan("  ")
    with pytest.raises(ValueError):
        experiment.promote(plan="  ")
    running = experiment.start()
    with pytest.raises(ValueError):
        running.abandon("  ")


def test_failed_run_releases_only_a_row_it_started(tmp_path, monkeypatch):
    import importlib

    evaluate_module = importlib.import_module("skore._dispatch.evaluate")
    real_evaluate = evaluate_module.evaluate
    project = _project(tmp_path)
    backlog = create_experiment(project, "backlog", "question", status="backlog")
    running = create_experiment(
        project, "running", "question", status="backlog"
    ).start()

    def fail(*args, **kwargs):
        raise RuntimeError("evaluate failed")

    monkeypatch.setattr(evaluate_module, "evaluate", fail)
    with pytest.raises(RuntimeError, match="evaluate failed"):
        backlog.run(Ridge(), *_data())
    assert project.experiments.get(backlog.id).status == "backlog"

    with pytest.raises(RuntimeError, match="evaluate failed"):
        running.run(Ridge(), *_data())
    assert project.experiments.get(running.id).status == "running"

    monkeypatch.setattr(
        evaluate_module,
        "evaluate",
        lambda *args, **kwargs: real_evaluate(Ridge(random_state=0), *_data()),
    )

    def fail_put(key, report):
        raise RuntimeError("store failed")

    monkeypatch.setattr(project, "put", fail_put)
    fresh = create_experiment(project, "stored", "question", status="backlog")
    with pytest.raises(RuntimeError, match="store failed"):
        fresh.run(Ridge(), *_data())
    assert project.experiments.get(fresh.id).status == "backlog"

    monkeypatch.setattr(evaluate_module, "evaluate", lambda *args, **kwargs: "nope")
    typed = create_experiment(project, "typed", "question", status="backlog")
    with pytest.raises(TypeError):
        typed.run(Ridge(), *_data())
    assert project.experiments.get(typed.id).status == "backlog"


def test_put_creates_no_experiments(tmp_path):
    from skore import evaluate

    project = _project(tmp_path)
    project.put("report", evaluate(Ridge(random_state=0), *_data()))
    assert project.experiments.frame.empty
    with pytest.raises(KeyError):
        project.experiments.get("missing")


def test_frame_is_ordered_by_creation_and_reread(tmp_path):
    project = _project(tmp_path)
    assert list(project.experiments.frame.columns) == [
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
    ]
    first = create_experiment(project, "first", "one")
    assert project.experiments.frame.empty is False
    second = create_experiment(project, "second", "two")
    assert project.experiments.frame["stem"].tolist() == ["first", "second"]
    assert project.experiments.get(first.id).stem == "first"
    assert project.experiments.get(second.id).created_at >= first.created_at


def test_concurrent_start(tmp_path):
    project = _project(tmp_path)
    experiment = create_experiment(project, "stem", "question", status="backlog")
    payload = json.dumps(experiment._row)
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", _CHILD, str(tmp_path), project.name, payload],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    outputs = []
    for process in processes:
        stdout, stderr = process.communicate()
        assert process.returncode == 0, stderr
        outputs.append(stdout.strip().splitlines()[-1])
    assert sorted(outputs) == ["conflict", "ok"]
    stored = project.experiments.get(experiment.id)
    assert stored.status == "running"
    assert stored.version == 2


def test_storage_rejects_a_bad_row(tmp_path):
    project = _project(tmp_path)
    experiment = create_experiment(project, "stem", "question")
    assert experiment.updated_at

    with pytest.raises(JournalConflictError, match="already exists"):
        project._create_experiment(experiment._row)
    with pytest.raises(KeyError):
        project._update_experiment({**experiment._row, "id": "missing"})
    with pytest.raises(ValueError, match="Cannot move"):
        project._update_experiment({**experiment._row, "status": "running"})
    with pytest.raises(ValueError, match="Unknown experiment status"):
        project._update_experiment({**experiment._row, "status": "nope"})

    running = create_experiment(
        project, "running", "question", status="backlog"
    ).start()
    with pytest.raises(ValueError, match="reason"):
        project._update_experiment({**running._row, "status": "abandoned"})


def test_failed_write_removes_the_temporary_file(tmp_path, monkeypatch):
    project = _project(tmp_path)
    experiment = create_experiment(project, "stem", "question")
    directory = project._Project__project.path / "experiments"

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr("skore._plugins.local.project.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        experiment.update_plan("next")
    assert list(directory.glob(".*.tmp")) == []
    assert project.experiments.get(experiment.id).plan is None

    def remove_temp(*args, **kwargs):
        descriptor, temporary = mkstemp(*args, **kwargs)
        os.close(descriptor)
        os.unlink(temporary)
        return descriptor, temporary

    monkeypatch.setattr("skore._plugins.local.project.mkstemp", remove_temp)
    with pytest.raises(OSError):
        experiment.update_plan("next")
    assert project.experiments.get(experiment.id).plan is None
