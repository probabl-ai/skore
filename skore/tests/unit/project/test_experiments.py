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
