"""Tests for the ``skore sync`` command."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from skore._cli.sync import commands as _commands
from skore._cli.sync.commands import sync

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _plain_output(output: str) -> str:
    return _ANSI_ESCAPE.sub("", output)


def test_sync_help_describes_endpoints():
    result = CliRunner().invoke(sync, ["--help"])
    output = _plain_output(result.output)
    option_help = {
        parameter.name: parameter.help
        for parameter in sync.params
        if hasattr(parameter, "help")
    }

    assert result.exit_code == 0
    assert "``SOURCE_PROJECT`` is the source project name." in output
    assert option_help["from_mode"] == (
        "Storage mode of the source project. Defaults to ``local``."
    )
    assert option_help["from_workspace"].startswith(
        "Source workspace. Required for ``hub``"
    )
    assert option_help["to_project"] == (
        "Destination project name. Defaults to ``SOURCE_PROJECT``."
    )
    assert option_help["to_workspace"].startswith(
        "Destination workspace. Required for ``hub``"
    )
    assert option_help["tracking_uri"].startswith(
        "MLflow tracking URI. Applied to every MLflow endpoint"
    )


@dataclass
class _Project:
    name: str
    mode: str
    workspace: object | None = None
    tracking_uri: str | None = None

    def sync(self, other, *, bidirectional, dry_run):
        self.sync_args = (other, bidirectional, dry_run)
        return pd.DataFrame(
            {"key": ["report"], "direction": ["outbound"], "status": ["planned"]},
            index=pd.Index(["report-id"], name="report_id"),
        )


@pytest.fixture
def projects(monkeypatch):
    created = []

    def project(name, *, mode, **kwargs):
        instance = _Project(name=name, mode=mode, **kwargs)
        created.append(instance)
        return instance

    monkeypatch.setattr("skore.Project", project)
    return created


def test_sync_defaults_source_to_local_and_forwards_options(projects):
    created = projects

    result = CliRunner().invoke(
        sync,
        ["experiment", "--to=hub", "--to-workspace=team", "--both", "--dry-run"],
        env={"SKORE_HUB_API_KEY": "key"},
    )

    assert result.exit_code == 0, result.output
    assert [(project.name, project.mode, project.workspace) for project in created] == [
        ("experiment", "local", None),
        ("experiment", "hub", "team"),
    ]
    assert created[0].sync_args == (created[1], True, True)
    assert "Dry run complete" in result.output


def test_sync_defaults_destination_to_local_with_custom_name(projects):
    created = projects

    result = CliRunner().invoke(
        sync,
        [
            "production",
            "--from=hub",
            "--from-workspace=team",
            "--to-project=downloaded",
            "--to-workspace=~/skore",
        ],
        env={"SKORE_HUB_API_KEY": "key"},
    )

    assert result.exit_code == 0, result.output
    assert [(project.name, project.mode, project.workspace) for project in created] == [
        ("production", "hub", "team"),
        ("downloaded", "local", Path("~/skore").expanduser().resolve()),
    ]


def test_sync_uses_one_mlflow_tracking_uri(projects):
    created = projects

    result = CliRunner().invoke(
        sync,
        [
            "source",
            "--from=mlflow",
            "--to=mlflow",
            "--to-project=destination",
            "--tracking-uri=http://mlflow.test",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [project.tracking_uri for project in created] == [
        "http://mlflow.test",
        "http://mlflow.test",
    ]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["experiment"], "Pass --from or --to"),
        (["experiment", "--to=hub"], "--to-workspace is required for Hub"),
        (
            ["experiment", "--to=mlflow", "--to-workspace=workspace"],
            "--to-workspace is not valid for MLflow",
        ),
        (
            ["experiment", "--from=mlflow", "--to=local", "--host=http://hub.test"],
            "--host requires a Hub endpoint",
        ),
        (
            ["experiment", "--from=local", "--tracking-uri=http://mlflow.test"],
            "--tracking-uri requires an MLflow endpoint",
        ),
    ],
)
def test_sync_validates_options(args, message):
    result = CliRunner().invoke(sync, args)

    assert result.exit_code == 2
    assert message in _plain_output(result.output)


def test_sync_rejects_same_endpoint_before_construction(projects):
    created = projects

    result = CliRunner().invoke(sync, ["experiment", "--from=local", "--to=local"])

    assert result.exit_code == 2
    assert "same project" in _plain_output(result.output)
    assert created == []


def test_sync_host_resolves_before_project_construction(projects, monkeypatch):
    seen = []
    monkeypatch.setattr(
        _commands,
        "resolve_hub_uri",
        lambda url: seen.append(url) or url,
    )

    result = CliRunner().invoke(
        sync,
        [
            "experiment",
            "--to=hub",
            "--to-workspace=team",
            "--host=https://frontend.test",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen == ["https://frontend.test"]


def test_sync_hub_does_not_require_env_api_key(projects):
    created = projects

    result = CliRunner().invoke(
        sync,
        ["experiment", "--to=hub", "--to-workspace=team"],
    )

    assert result.exit_code == 0, result.output
    assert [(project.name, project.mode, project.workspace) for project in created] == [
        ("experiment", "local", None),
        ("experiment", "hub", "team"),
    ]


def test_sync_reports_empty_result(capsys):
    _commands._render_result(pd.DataFrame(), dry_run=False)

    assert capsys.readouterr().out == "No reports to synchronize.\n"


def test_sync_converts_backend_errors(monkeypatch):
    def project(*args, **kwargs):
        raise RuntimeError("backend error")

    monkeypatch.setattr("skore.Project", project)

    result = CliRunner().invoke(sync, ["experiment", "--to=mlflow"])

    assert result.exit_code != 0
    assert "backend error" in result.output
