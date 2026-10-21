"""Tests for the experiment collection and its lifecycle."""

import pytest
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor

from skore import EstimatorReport, Project, evaluate


def test_create_run_and_check_ideas(tmp_path):
    project = Project("housing", mode="local", workspace=tmp_path)
    project.objective = "Predict the target. Metric: RMSE."

    from skore.agent import create_experiment

    experiment = create_experiment(
        project,
        stem="01_dummy",
        question="What RMSE does a constant predictor reach?",
        plan="DummyRegressor, 2-fold CV.",
    )
    assert experiment.status == "backlog"
    X, y = make_regression(n_samples=20, n_features=3, random_state=0)
    report = experiment.run(DummyRegressor(), X, y, splitter=2)

    assert report.id == experiment.id
    assert project.get(experiment.id).id == experiment.id
    assert project.experiments.get(experiment.id).status == "history"
    with pytest.raises(ValueError, match="only once"):
        project.experiments.get(experiment.id).run(DummyRegressor(), X, y, splitter=2)

    ideas = project.experiments.frame.query("check_code == 'SKD004'")
    assert ideas.empty or set(ideas["status"]) <= {
        "idea",
        "aside",
        "discarded",
        "backlog",
        "history",
    }


def test_decide_promote_and_invalid_transitions(tmp_path):
    from skore.agent import create_experiment

    project = Project("housing", mode="local", workspace=tmp_path)
    idea = create_experiment(project, stem="02_idea", question="Try a new feature?")
    assert idea.status == "idea"
    idea.decide("aside")
    assert project.experiments.get(idea.id).status == "aside"
    with pytest.raises(ValueError, match="Decide it back"):
        project.experiments.get(idea.id).promote("A plan.")
    project.experiments.get(idea.id).decide("idea")
    promoted = project.experiments.get(idea.id).promote("Add the feature.")
    assert promoted.status == "backlog"
    assert promoted.plan == "Add the feature."
    with pytest.raises(ValueError, match="Cannot decide"):
        promoted.decide("idea")


def test_put_refreshes_an_idea_and_keeps_a_discarded_check(tmp_path, monkeypatch):
    project = Project("housing", mode="local", workspace=tmp_path)
    X, y = make_regression(n_samples=20, n_features=3, random_state=0)
    report = evaluate(DummyRegressor(), X, y, splitter=2)
    report._check_results_cache = {
        "SKD004": {
            "title": "First wording",
            "section": "issue",
            "docs_url": "",
            "explanation": "",
        }
    }
    monkeypatch.setattr(
        EstimatorReport,
        "_get_checks_results",
        lambda self, ignored_codes, fast_mode=False: getattr(
            self, "_check_results_cache", {}
        ),
    )
    project.put("01_dummy", report)
    ideas = project.experiments.frame.query("check_code == 'SKD004'")
    assert len(ideas) == 1
    assert ideas.iloc[0]["status"] == "idea"
    assert ideas.iloc[0]["question"] == "First wording"

    idea = project.experiments.get(ideas.iloc[0]["id"])
    idea.decide("discarded")
    report._check_results_cache["SKD004"]["title"] = "Second wording"
    project.put("01_dummy", report)
    again = project.experiments.frame.query("check_code == 'SKD004'")
    assert len(again) == 1
    assert again.iloc[0]["status"] == "discarded"
    assert again.iloc[0]["question"] == "First wording"


def test_journal_renders_without_loading_reports(tmp_path):
    from skore.agent import create_experiment

    project = Project("housing", mode="local", workspace=tmp_path)
    project.objective = "Predict the target."
    create_experiment(project, stem="03_user", question="What if we drop outliers?")
    markdown = project.summarize().to_markdown()
    assert "Predict the target." in markdown
    assert "What if we drop outliers?" in markdown
    assert "open" in markdown
    assert not hasattr(project.summarize(), "experiments")


def test_sync_does_not_rewind_history(tmp_path):
    from skore.agent import create_experiment

    source = Project("housing", mode="local", workspace=tmp_path / "source")
    target = Project("housing", mode="local", workspace=tmp_path / "target")
    experiment = create_experiment(
        source,
        stem="01_dummy",
        question="Older.",
        plan="Older plan.",
    )
    target._write_experiment(
        {
            **experiment._row,
            "status": "history",
            "question": "Kept.",
            "plan": "Kept plan.",
        }
    )
    source.sync(target)
    kept = target.experiments.get(experiment.id)
    assert kept.status == "history"
    assert kept.question == "Kept."


def test_rejected_rows_and_duplicate_experiments(tmp_path):
    from skore._project.experiments import (
        ensure_report_id,
        normalize_row,
        transition_allowed,
    )
    from skore.agent import create_experiment

    project = Project("housing", mode="local", workspace=tmp_path)
    with pytest.raises(ValueError, match="Unknown experiment status"):
        normalize_row(
            {
                "id": "1",
                "stem": "01",
                "status": "nope",
                "question": "Q?",
            }
        )
    with pytest.raises(ValueError, match="stem is required"):
        normalize_row({"id": "1", "stem": "  ", "status": "idea", "question": "Q?"})
    with pytest.raises(ValueError, match="question is required"):
        normalize_row({"id": "1", "stem": "01", "status": "idea", "question": " "})
    assert transition_allowed(None, "nope", None) is False
    assert transition_allowed("history", "idea", None) is False

    first = create_experiment(project, stem="01_dummy", question="First?")
    with pytest.raises(ValueError, match="already exists"):
        create_experiment(project, stem="01_dummy", question="Again?")
    create_experiment(
        project,
        stem="02_check",
        question="A check.",
        source_id=first.id,
        check_code="SKD004",
    )
    with pytest.raises(ValueError, match="already exists"):
        create_experiment(
            project,
            stem="03_check",
            question="Same check.",
            source_id=first.id,
            check_code="SKD004",
        )
    assert first.source_id is None
    assert first.check_code is None
    with pytest.raises(KeyError):
        project.experiments.get("missing")
    with pytest.raises(ValueError, match="requires a plan"):
        project.experiments.get(first.id).promote("  ")
    with pytest.raises(ValueError, match="backlog"):
        project.experiments.get(first.id).run(DummyRegressor())
    with pytest.raises(ValueError, match="Cannot move"):
        project._write_experiment({**first._row, "status": "history"})
    with pytest.raises(TypeError, match="string"):
        project.objective = None

    report = evaluate(
        DummyRegressor(),
        *make_regression(n_samples=20, n_features=3, random_state=0),
        splitter=2,
    )
    report._metadata["id"] = "kept"
    with pytest.raises(ValueError, match="does not match"):
        ensure_report_id(report, "other")
    report.reports_ = (report,)
    assert ensure_report_id(report) == "kept"


def test_run_refuses_a_second_report_and_a_non_report(tmp_path, monkeypatch):
    from skore.agent import create_experiment

    project = Project("housing", mode="local", workspace=tmp_path)
    experiment = create_experiment(
        project, stem="01_dummy", question="What RMSE?", plan="A plan."
    )
    X, y = make_regression(n_samples=20, n_features=3, random_state=0)
    stored = evaluate(DummyRegressor(), X, y, splitter=2)
    stored._metadata["id"] = experiment.id
    project.put("01_dummy", stored)
    with pytest.raises(ValueError, match="only once"):
        experiment.run(DummyRegressor(), X, y, splitter=2)

    monkeypatch.setattr("skore.evaluate", lambda *args, **kwargs: object())
    fresh = create_experiment(
        project, stem="02_other", question="What else?", plan="Another plan."
    )
    with pytest.raises(TypeError, match="estimator or cross-validation"):
        fresh.run(DummyRegressor(), X, y)


def test_journal_renders_sources_headlines_and_selected_reports(tmp_path):
    from pandas import DataFrame, MultiIndex

    from skore._project.journal import render_journal

    def pd_index(ids: list[str]) -> MultiIndex:
        return MultiIndex.from_arrays([ids], names=["id"])

    experiments = DataFrame(
        [
            {
                "id": "parent",
                "stem": "01_dummy",
                "status": "history",
                "question": "What RMSE?",
                "source_id": None,
                "check_code": None,
                "plan": "A plan.",
            },
            {
                "id": "child",
                "stem": "02_follow",
                "status": "idea",
                "question": "Drop a feature?",
                "source_id": "parent",
                "check_code": None,
                "plan": None,
            },
            {
                "id": "audit",
                "stem": "03_audit",
                "status": "backlog",
                "question": "Class imbalance.",
                "source_id": "missing-parent",
                "check_code": "SKD004",
                "plan": "Check it.",
            },
        ]
    )
    summary = DataFrame(
        {"rmse": [1.25], "accuracy_std": [0.1], "note": ["kept"]},
        index=pd_index(["parent"]),
    )
    text = render_journal(
        "Predict the target.",
        experiments,
        summary,
        report_markdown={"parent": "# Report", "other": "# Hidden"},
    )
    assert "design:01_dummy" in text
    assert "audit:missing-parent:checks.SKD004" in text
    assert "rmse=1.25" in text
    assert "# Report" in text
    assert "# Hidden" not in text
    without_summary = render_journal("Objective.", experiments, DataFrame())
    assert "rmse=" not in without_summary
    other = DataFrame(
        {"accuracy": [0.5], "accuracy_std": [0.1], "blank": [float("nan")]},
        index=summary.index,
    )
    assert "accuracy=0.5" in render_journal("Objective.", experiments.iloc[:1], other)


def test_experiment_rows_ignore_a_backend_without_a_list(tmp_path, monkeypatch):
    project = Project("housing", mode="local", workspace=tmp_path)
    backend = project._Project__project
    monkeypatch.setattr(backend, "experiments", lambda: {"not": "a list"})
    assert project.experiments.frame.empty
    monkeypatch.setattr(backend, "experiments", None)
    assert project.experiments.frame.empty


def test_check_ideas_tolerate_a_backend_that_cannot_store_them(tmp_path, monkeypatch):
    project = Project("housing", mode="local", workspace=tmp_path)
    X, y = make_regression(n_samples=20, n_features=3, random_state=0)
    report = evaluate(DummyRegressor(), X, y)
    report._metadata.pop("id", None)

    class BrokenChecks:
        def summarize(self, fast_mode=False):
            raise RuntimeError("checks unavailable")

    report._check_results_cache = {}
    report.checks = BrokenChecks()
    project._raise_check_ideas("01_dummy", report)
    project._raise_check_ideas("01_dummy", report)

    report._metadata["id"] = "report-1"
    report._check_results_cache = {
        "SKD001": "not-a-dict",
        "SKD002": {"section": "tip", "title": ""},
        "SKD004": {"section": "issue", "title": "Imbalance."},
    }
    from skore.agent import create_experiment

    create_experiment(project, stem="01_dummy-skd004", question="Taken stem.")
    project._raise_check_ideas("01_dummy", report)
    created = project.experiments.frame.query("check_code == 'SKD004'")
    assert created.iloc[0]["stem"] == "01_dummy-skd004-2"

    def fail_write(row):
        raise RuntimeError("backend down")

    monkeypatch.setattr(project, "_write_experiment", fail_write)
    report._check_results_cache = {
        "SKD004": {"section": "issue", "title": "Updated wording."}
    }
    project._raise_check_ideas("01_dummy", report)
    report._check_results_cache = {
        "SKD009": {"section": "issue", "title": "Slower than baseline."}
    }
    project._raise_check_ideas("01_dummy", report)

    def fail_rows():
        raise RuntimeError("frame unavailable")

    monkeypatch.setattr(project, "_experiment_rows", fail_rows)
    project._raise_check_ideas("01_dummy", report)


def test_summary_can_embed_report_markdown(tmp_path):
    project = Project("housing", mode="local", workspace=tmp_path)
    project.objective = "Predict the target."
    X, y = make_regression(n_samples=20, n_features=3, random_state=0)
    project.put("01_dummy", evaluate(DummyRegressor(), X, y))
    text = project.summarize().to_markdown(with_reports=True)
    assert "Predict the target." in text
    assert "DummyRegressor" in text or "01_dummy" in text


def test_sync_copies_a_non_empty_objective(tmp_path):
    source = Project("housing", mode="local", workspace=tmp_path / "source")
    target = Project("housing", mode="local", workspace=tmp_path / "target")
    source.objective = "Predict the target."
    source.sync(target)
    assert target.objective == "Predict the target."


def test_package_getattr_exposes_the_agent_module():
    import skore
    import skore._project.protocol  # noqa: F401

    assert skore.__getattr__("agent").create_experiment
    assert skore.__getattr__("Project") is skore.Project
