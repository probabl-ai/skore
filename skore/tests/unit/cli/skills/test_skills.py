import json
import re

from click.testing import CliRunner
from textual.widgets import SelectionList, TabbedContent

from skore._cli import cli
from skore._cli.app.help import HelpInput, HelpScreen
from skore._cli.skills import commands as _skills
from skore._cli.skills import release as skills_release
from skore._cli.skills import render as _render
from skore._cli.skills.app.widgets import AutoRadioSet
from skore._cli.skills.catalog import GITHUB_REPO, fetch_release
from skore._cli.skills.commands import (
    ProbablSkillsInstaller,
)

SIDECAR = ".skore-skill.json"
LOCAL_CATALOG = ".catalog.json"
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

GAMMA_CATALOG = {
    "skills": [
        {
            "id": "gamma",
            "path": "skills/gamma",
            "title": "Gamma",
            "summary": "The gamma skill",
            "category": "tooling",
            "hash": "hash-gamma-1",
        }
    ],
    "workflows": [],
}


def _invoke(args, **kwargs):
    return CliRunner().invoke(cli, args, **kwargs)


def _register_gamma(release, tag="9.0.0"):
    """Serve a second catalog under ``acme/skills`` and return it."""
    catalog = json.loads(json.dumps(GAMMA_CATALOG))
    release["by_repo"]["acme/skills"] = {"tag": tag, "catalog": catalog}
    return catalog


def _plain_output(output: str) -> str:
    """Strip ANSI codes from rich-click error panels for stable assertions."""
    return _ANSI_ESCAPE.sub("", output)


async def _wait_wizard_step(app, pilot, step_id: str) -> None:
    wizard = app.query_one("#wizard", TabbedContent)
    for _ in range(50):
        await pilot.pause()
        if wizard.active != step_id:
            continue
        if step_id == "step-agents":
            radio = app.query_one("#agents", AutoRadioSet)
            if radio.has_focus and radio.pressed_index >= 0:
                return
        elif step_id == "step-scope":
            radio = app.query_one("#scope", AutoRadioSet)
            if radio.has_focus and radio.pressed_index >= 0:
                return
        elif step_id == "step-skills":
            # Switching sources remounts the lists, so wait for the single
            # remaining list to match the catalog that was just fetched.
            lists = app.query("#sel-skills")
            if len(lists) != 1 or app.catalog is None:
                continue
            expected = {skill["id"] for skill in app.catalog["skills"]}
            if {option.value for option in lists.first(SelectionList).options} != (
                expected
            ):
                continue
            return
        else:
            return
    raise AssertionError(f"wizard step {step_id!r} not ready")


async def _confirm_source(app, pilot, repo: str | None = None) -> None:
    """Confirm the source step, optionally overriding the GitHub repo."""
    if repo is not None:
        app.query_one("#repo", HelpInput).value = repo
    await pilot.press("enter")
    await _wait_wizard_step(app, pilot, "step-skills")


async def _back_to_source(app, pilot) -> None:
    """Reopen the source step of an already-confirmed wizard by clicking its tab."""
    await pilot.click("#--content-tab-step-source")
    await _wait_wizard_step(app, pilot, "step-source")


async def _wait_workflow_skills_sync(
    app, pilot, expected: set[str], *, max_attempts: int = 50
) -> None:
    """Wait until workflow-driven skill selection matches ``expected``."""
    skill_list = app.query_one("#sel-skills", SelectionList)
    for _ in range(max_attempts):
        await pilot.pause(delay=0)
        if set(skill_list.selected) == expected:
            return
    raise AssertionError(
        f"expected skills {expected!r}, got {set(skill_list.selected)!r}"
    )


def test_install_skill_project(release, workspace):
    result = _invoke(["skills", "install", "alpha"])

    assert result.exit_code == 0
    skill_dir = workspace.project / ".agents" / "skills" / "alpha"
    assert (skill_dir / "SKILL.md").is_file()

    sidecar = json.loads((skill_dir / SIDECAR).read_text())
    assert sidecar == {
        "id": "alpha",
        "release": "0.1.0",
        "hash": "hash-alpha-1",
        "repository": "probabl-ai/skore-skills",
    }
    local_catalog = json.loads(
        (workspace.project / ".agents" / "skills" / ".catalog.json").read_text()
    )
    assert "probabl-ai/skore-skills" in local_catalog["sources"]
    assert local_catalog["sources"]["probabl-ai/skore-skills"]["release"] == "0.1.0"
    assert local_catalog["sources"]["probabl-ai/skore-skills"]["version"] == "0.1.0"


def test_install_workflow_expands_to_skills(release, workspace):
    result = _invoke(["skills", "install", "flow"])

    assert result.exit_code == 0
    skills_dir = workspace.project / ".agents" / "skills"
    assert (skills_dir / "alpha" / "SKILL.md").is_file()
    assert (skills_dir / "beta" / "SKILL.md").is_file()


def test_install_unknown_identifier(release, workspace):
    result = _invoke(["skills", "install", "does-not-exist"])

    assert result.exit_code != 0
    assert "Unknown skill or workflow" in result.output


def test_install_explicit_agent(release, workspace):
    result = _invoke(["skills", "install", "alpha", "-a", "cursor"])

    assert result.exit_code == 0
    assert (workspace.project / ".cursor" / "skills" / "alpha").is_dir()
    assert not (workspace.project / ".agents").exists()


def test_install_bob_agent(release, workspace):
    result = _invoke(["skills", "install", "alpha", "-a", "bob"])

    assert result.exit_code == 0
    assert (workspace.project / ".bob" / "skills" / "alpha").is_dir()
    assert not (workspace.project / ".agents").exists()


def test_install_windsurf_global(release, workspace):
    result = _invoke(["skills", "install", "alpha", "-a", "windsurf", "-g"])

    assert result.exit_code == 0
    assert (workspace.home / ".codeium" / "windsurf" / "skills" / "alpha").is_dir()
    assert not (workspace.project / ".windsurf").exists()


def test_install_copilot_global_errors(release, workspace):
    result = _invoke(["skills", "install", "alpha", "-a", "github-copilot", "-g"])

    assert result.exit_code != 0
    assert "Copilot in VSCode has no user-level skills directory" in _plain_output(
        result.output
    )
    assert not (workspace.home / ".github").exists()


def test_install_global_scope(release, workspace):
    result = _invoke(["skills", "install", "alpha", "-g"])

    assert result.exit_code == 0
    assert (workspace.home / ".agents" / "skills" / "alpha").is_dir()
    assert not (workspace.project / ".agents").exists()


def test_install_all_without_ids_installs_everything(release, workspace):
    result = _invoke(["skills", "install", "--all"])

    assert result.exit_code == 0
    skills_dir = workspace.project / ".agents" / "skills"
    assert (skills_dir / "alpha" / "SKILL.md").is_file()
    assert (skills_dir / "beta" / "SKILL.md").is_file()


def test_install_agent_without_selection_errors(release, workspace):
    result = _invoke(["skills", "install", "-a", "cursor"])

    assert result.exit_code != 0
    assert "non-interactively" in result.output
    assert not (workspace.project / ".cursor").exists()


def test_render_catalog_interactive(monkeypatch):
    rendered = []
    catalog = {
        "workflows": [{"id": "flow", "summary": "A workflow"}],
        "skills": [{"id": "alpha", "summary": "A skill"}],
    }

    monkeypatch.setattr(_render, "is_non_interactive", lambda: False)
    monkeypatch.setattr(_render.console, "print", rendered.append)

    _render.render_catalog(catalog)

    assert [table.title for table in rendered] == ["Workflows (recommended)", "Skills"]


def test_render_catalog_filters_by_id_and_query(monkeypatch):
    catalog = {
        "workflows": [{"id": "flow", "summary": "A workflow", "title": "Flow"}],
        "skills": [
            {
                "id": "alpha",
                "title": "Alpha",
                "summary": "First skill",
                "category": "tooling",
            },
            {
                "id": "beta",
                "title": "Beta",
                "summary": "Second skill",
                "category": "reporting",
            },
        ],
    }
    rendered = []
    monkeypatch.setattr(_render, "is_non_interactive", lambda: False)
    monkeypatch.setattr(_render.console, "print", rendered.append)

    _render.render_catalog(catalog, ids=["alpha"])
    assert list(rendered[-1].columns[0].cells) == ["alpha"]

    rendered.clear()
    _render.render_catalog(catalog, query="reporting")
    assert list(rendered[-1].columns[0].cells) == ["beta"]

    rendered.clear()
    _render.render_catalog(catalog, query="missing")
    assert rendered == ["No skill or workflow matches the query."]


def test_render_catalog_plain_text(monkeypatch):
    rendered = []
    monkeypatch.setattr(_render, "is_non_interactive", lambda: True)
    monkeypatch.setattr(_render.console, "print", rendered.append)
    catalog = {
        "workflows": [{"id": "flow", "summary": "A workflow"}],
        "skills": [{"id": "alpha", "summary": "A skill"}],
    }

    _render.render_catalog(catalog)
    assert rendered[0] == "Workflows:"
    assert any(line == "Skills:" for line in rendered)

    rendered.clear()
    _render.render_catalog({"skills": catalog["skills"]})
    assert rendered[0] == "Skills:"
    assert "" not in rendered

    rendered.clear()
    _render.render_catalog({"workflows": [], "skills": []})
    assert rendered == ["No skill or workflow matches the query."]


def test_interactive_install_cleans_up_when_dismissed(monkeypatch, tmp_path):
    extracted = tmp_path / "extract" / "repo"
    extracted.mkdir(parents=True)

    class _Dismissed:
        result = None

        def __init__(self, **kwargs):
            self.root = extracted

        def run(self):
            return None

    monkeypatch.setattr(_skills, "ProbablSkillsInstaller", _Dismissed)

    assert (
        _skills._interactive_install_options(
            agent=(), default_global=False, default_repo="owner/name"
        )
        is None
    )
    assert not extracted.exists()


def test_install_global_without_selection_errors(release, workspace):
    result = _invoke(["skills", "install", "-g"])

    assert result.exit_code != 0
    assert not (workspace.home / ".agents").exists()


def test_install_interactive_selection(release, workspace, monkeypatch):
    monkeypatch.setattr(_skills, "is_non_interactive", lambda: False)

    def fake_options(*, agent, default_global, default_repo):
        tag, root, catalog = fetch_release(default_repo)
        return (
            [_skills._index(catalog)[0]["alpha"]],
            ["agents"],
            False,
            default_repo,
            tag,
            root,
            catalog,
        )

    monkeypatch.setattr(_skills, "_interactive_install_options", fake_options)

    result = _invoke(["skills", "install"])

    assert result.exit_code == 0
    skills_dir = workspace.project / ".agents" / "skills"
    assert (skills_dir / "alpha" / "SKILL.md").is_file()
    assert not (skills_dir / "beta").exists()


def test_install_interactive_agent_and_global(release, workspace, monkeypatch):
    monkeypatch.setattr(_skills, "is_non_interactive", lambda: False)

    def fake_options(*, agent, default_global, default_repo):
        tag, root, catalog = fetch_release(default_repo)
        return (
            [_skills._index(catalog)[0]["alpha"]],
            ["cursor"],
            True,
            default_repo,
            tag,
            root,
            catalog,
        )

    monkeypatch.setattr(_skills, "_interactive_install_options", fake_options)

    result = _invoke(["skills", "install"])

    assert result.exit_code == 0
    assert (workspace.home / ".cursor" / "skills" / "alpha").is_dir()
    assert not (workspace.project / ".cursor").exists()


def test_install_interactive_cancelled(release, workspace, monkeypatch):
    monkeypatch.setattr(_skills, "is_non_interactive", lambda: False)
    monkeypatch.setattr(
        _skills,
        "_interactive_install_options",
        lambda *, agent, default_global, default_repo: None,
    )

    result = _invoke(["skills", "install"])

    assert result.exit_code == 0
    assert "Nothing selected" in result.output
    assert not (workspace.project / ".agents").exists()


def _fake_app(result, *, catalog, tag="0.1.0", root):
    class _FakeApp:
        def __init__(self, *, agent, default_global, default_repo):
            self.result = result
            self._catalog = catalog
            self._tag = tag
            self._root = root

        @property
        def catalog(self):
            return self._catalog

        @property
        def tag(self):
            return self._tag

        @property
        def root(self):
            return self._root

        def run(self):
            return None

    return _FakeApp


def _fake_manage_picker(result):
    class _FakePicker:
        def __init__(self, skill_ids, *, title, sources=None):
            self.result = result

        def run(self):
            return None

    return _FakePicker


def test_interactive_options_expands_workflow(release, monkeypatch):
    tag, root, catalog = fetch_release()
    monkeypatch.setattr(
        _skills,
        "ProbablSkillsInstaller",
        _fake_app(
            (["flow"], ["agents"], False, GITHUB_REPO),
            catalog=catalog,
            tag=tag,
            root=root,
        ),
    )

    selected, agents, global_, repo, out_tag, out_root, out_catalog = (
        _skills._interactive_install_options(
            agent=(), default_global=False, default_repo=GITHUB_REPO
        )
    )

    assert {skill["id"] for skill in selected} == {"alpha", "beta"}
    assert agents == ["agents"]
    assert global_ is False
    assert repo == GITHUB_REPO
    assert out_tag == tag
    assert out_root == root
    assert out_catalog == catalog


def test_interactive_options_individual_global(release, monkeypatch):
    tag, root, catalog = fetch_release()
    monkeypatch.setattr(
        _skills,
        "ProbablSkillsInstaller",
        _fake_app(
            (["beta"], ["cursor"], True, GITHUB_REPO),
            catalog=catalog,
            tag=tag,
            root=root,
        ),
    )

    selected, agents, global_, repo, _, _, _ = _skills._interactive_install_options(
        agent=(), default_global=False, default_repo=GITHUB_REPO
    )

    assert {skill["id"] for skill in selected} == {"beta"}
    assert agents == ["cursor"]
    assert global_ is True
    assert repo == GITHUB_REPO


def test_interactive_options_cancelled(release, monkeypatch):
    tag, root, catalog = fetch_release()
    monkeypatch.setattr(
        _skills,
        "ProbablSkillsInstaller",
        _fake_app(None, catalog=catalog, tag=tag, root=root),
    )

    assert (
        _skills._interactive_install_options(
            agent=(), default_global=False, default_repo=GITHUB_REPO
        )
        is None
    )


def test_interactive_options_empty_selection(release, monkeypatch):
    tag, root, catalog = fetch_release()
    monkeypatch.setattr(
        _skills,
        "ProbablSkillsInstaller",
        _fake_app(
            ([], ["agents"], False, GITHUB_REPO),
            catalog=catalog,
            tag=tag,
            root=root,
        ),
    )

    assert (
        _skills._interactive_install_options(
            agent=(), default_global=False, default_repo=GITHUB_REPO
        )
        is None
    )


def test_interactive_options_without_release(release, monkeypatch):
    """A wizard that never fetched a release has nothing to clean up."""
    monkeypatch.setattr(
        _skills,
        "ProbablSkillsInstaller",
        _fake_app(
            (["alpha"], ["agents"], False, GITHUB_REPO),
            catalog=None,
            tag=None,
            root=None,
        ),
    )

    assert (
        _skills._interactive_install_options(
            agent=(), default_global=False, default_repo=GITHUB_REPO
        )
        is None
    )


async def test_wizard_app_full_flow(release):
    app = ProbablSkillsInstaller(agent=(), default_global=False)
    async with app.run_test() as pilot:
        await _confirm_source(app, pilot)
        app.query_one("#sel-workflows", SelectionList).select_all()
        await pilot.pause()
        await pilot.press("enter")  # confirm skills -> agents
        await _wait_wizard_step(app, pilot, "step-agents")
        await pilot.press("enter")  # confirm agents -> scope
        await _wait_wizard_step(app, pilot, "step-scope")
        await pilot.press("enter")  # confirm scope -> install
        await pilot.pause()

    selected_ids, agents, global_, repo = app.result

    assert set(selected_ids) == {"flow", "alpha", "beta"}
    assert agents == ["agents"]
    assert global_ is False
    assert repo == GITHUB_REPO


async def test_wizard_app_selecting_workflow_selects_its_skills(release):
    app = ProbablSkillsInstaller(agent=(), default_global=False)
    async with app.run_test() as pilot:
        await _confirm_source(app, pilot)
        app.query_one("#sel-workflows", SelectionList).select_all()
        await _wait_workflow_skills_sync(app, pilot, {"alpha", "beta"})
        selected = list(app.query_one("#sel-skills", SelectionList).selected)

    assert set(selected) == {"alpha", "beta"}


async def test_wizard_app_deselecting_workflow_deselects_its_skills(release):
    app = ProbablSkillsInstaller(agent=(), default_global=False)
    async with app.run_test() as pilot:
        await _confirm_source(app, pilot)
        workflows = app.query_one("#sel-workflows", SelectionList)
        workflows.select_all()
        await _wait_workflow_skills_sync(app, pilot, {"alpha", "beta"})
        workflows.deselect_all()
        await _wait_workflow_skills_sync(app, pilot, set())
        selected = list(app.query_one("#sel-skills", SelectionList).selected)

    assert selected == []


async def test_wizard_app_single_agent_choice(release):
    app = ProbablSkillsInstaller(agent=(), default_global=False)
    async with app.run_test() as pilot:
        await _confirm_source(app, pilot)
        app.query_one("#sel-workflows", SelectionList).select_all()
        await pilot.pause()
        await pilot.press("enter")  # confirm skills -> agents
        await _wait_wizard_step(app, pilot, "step-agents")
        await pilot.press("down")
        await pilot.pause()
        await pilot.press("enter")  # confirm agents -> scope
        await _wait_wizard_step(app, pilot, "step-scope")
        await pilot.press("enter")  # confirm scope -> install
        await pilot.pause()

    _, agents, _, _ = app.result

    assert agents == ["claude-code"]


async def test_wizard_app_skips_agent_step_when_provided(release):
    app = ProbablSkillsInstaller(agent=("cursor",), default_global=True)
    async with app.run_test() as pilot:
        await _confirm_source(app, pilot)
        app.query_one("#sel-skills", SelectionList).select_all()
        await pilot.pause()
        await pilot.press("enter")  # confirm skills -> scope (agents skipped)
        await _wait_wizard_step(app, pilot, "step-scope")
        await pilot.press("enter")  # confirm scope -> install
        await pilot.pause()

    selected_ids, agents, global_, repo = app.result

    assert set(selected_ids) == {"alpha", "beta"}
    assert agents == ["cursor"]
    assert global_ is True
    assert repo == GITHUB_REPO


async def test_wizard_app_cancel(release):
    app = ProbablSkillsInstaller(agent=(), default_global=False)
    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()

    assert app.result is None


async def test_wizard_app_requires_selection(release):
    app = ProbablSkillsInstaller(agent=(), default_global=False)
    async with app.run_test() as pilot:
        await _confirm_source(app, pilot)
        await pilot.press("enter")  # nothing selected -> stay on skills
        await pilot.pause()
        active = app.query_one("#wizard").active
        await pilot.press("escape")
        await pilot.pause()

    assert active == "step-skills"


async def test_wizard_app_rejects_invalid_repo(release):
    app = ProbablSkillsInstaller(agent=(), default_global=False)
    async with app.run_test() as pilot:
        app.query_one("#repo", HelpInput).value = "not-a-repo"
        await pilot.press("enter")
        await pilot.pause()
        active = app.query_one("#wizard").active
        await pilot.press("escape")
        await pilot.pause()

    assert active == "step-source"
    assert app.catalog is None


async def test_wizard_app_loads_custom_repo(release):
    other = {
        "skills": [
            {
                "id": "gamma",
                "path": "skills/gamma",
                "title": "Gamma",
                "summary": "The gamma skill",
                "category": "tooling",
                "hash": "hash-gamma-1",
            }
        ],
        "workflows": [],
    }
    release["by_repo"]["acme/skills"] = {"tag": "9.0.0", "catalog": other}

    app = ProbablSkillsInstaller(
        agent=("cursor",), default_global=False, default_repo=GITHUB_REPO
    )
    async with app.run_test() as pilot:
        await _confirm_source(app, pilot, "acme/skills")
        app.query_one("#sel-skills", SelectionList).select_all()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_wizard_step(app, pilot, "step-scope")
        await pilot.press("enter")
        await pilot.pause()

    selected_ids, agents, global_, repo = app.result

    assert selected_ids == ["gamma"]
    assert agents == ["cursor"]
    assert global_ is False
    assert repo == "acme/skills"
    assert app.tag == "9.0.0"


def test_list_reports_unreadable_sidecar(workspace):
    skill = workspace.project / ".agents" / "skills" / "alpha"
    skill.mkdir(parents=True)
    (skill / SIDECAR).write_text("[]\n")

    result = _invoke(["skills", "list"])

    assert result.exit_code != 0
    assert "object with an id" in result.output


def test_list_installed(release, workspace):
    _invoke(["skills", "install", "alpha"])
    result = _invoke(["skills", "list"])

    assert result.exit_code == 0
    assert "alpha" in result.output
    assert "probabl-ai/skore-skills" in result.output


def test_list_when_empty(release, workspace):
    result = _invoke(["skills", "list"])

    assert result.exit_code == 0
    assert "No skills installed" in result.output


def test_list_includes_non_default_agents(release, workspace):
    """Skills installed to a non-default agent are still listed by default."""
    _invoke(["skills", "install", "alpha", "-a", "cursor"])

    result = _invoke(["skills", "list"])

    assert result.exit_code == 0
    assert "alpha" in result.output


def test_list_global_skips_agents_without_user_directory(release, workspace):
    """A global scan of every agent skips Copilot instead of failing."""
    result = _invoke(["skills", "list", "-g"])

    assert result.exit_code == 0
    assert "No skills installed" in result.output


def test_list_copilot_global_errors(release, workspace):
    result = _invoke(["skills", "list", "-a", "github-copilot", "-g"])

    assert result.exit_code != 0
    assert "Copilot in VSCode has no user-level skills directory" in _plain_output(
        result.output
    )


def test_list_agent_restricts_scan(release, workspace):
    """``--agent`` narrows the scan to the requested agent only."""
    _invoke(["skills", "install", "alpha", "-a", "cursor"])

    result = _invoke(["skills", "list", "-a", "agents"])

    assert result.exit_code == 0
    assert "No skills installed" in result.output


def test_update_reinstalls_changed_skill(release, workspace):
    _invoke(["skills", "install", "alpha"])
    release["catalog"]["skills"][0]["hash"] = "hash-alpha-2"

    result = _invoke(["skills", "update", "--all"])

    assert result.exit_code == 0
    assert "updated" in result.output
    assert "probabl-ai/skore-skills" in result.output

    sidecar = json.loads(
        (workspace.project / ".agents" / "skills" / "alpha" / SIDECAR).read_text()
    )
    assert sidecar["hash"] == "hash-alpha-2"


def test_update_specific_id(release, workspace):
    _invoke(["skills", "install", "alpha", "beta"])
    release["catalog"]["skills"][0]["hash"] = "hash-alpha-2"
    release["catalog"]["skills"][1]["hash"] = "hash-beta-2"

    result = _invoke(["skills", "update", "alpha"])

    assert result.exit_code == 0
    assert "alpha" in result.output
    assert "beta" not in result.output


def test_update_when_up_to_date(release, workspace):
    _invoke(["skills", "install", "alpha"])

    result = _invoke(["skills", "update", "--all"])

    assert result.exit_code == 0
    assert "up to date" in result.output
    assert "probabl-ai/skore-skills" in result.output


def test_update_without_ids_errors(release, workspace):
    _invoke(["skills", "install", "alpha"])

    result = _invoke(["skills", "update"])

    assert result.exit_code != 0
    assert "Specify skill ids to update or pass --all" in _plain_output(result.output)


def test_update_non_default_agent(release, workspace):
    """A skill installed to a non-default agent is updated without ``--agent``."""
    _invoke(["skills", "install", "alpha", "-a", "cursor"])
    release["catalog"]["skills"][0]["hash"] = "hash-alpha-2"

    result = _invoke(["skills", "update", "--all"])

    assert result.exit_code == 0
    assert "updated" in result.output

    sidecar = json.loads(
        (workspace.project / ".cursor" / "skills" / "alpha" / SIDECAR).read_text()
    )
    assert sidecar["hash"] == "hash-alpha-2"


def test_remove_skill(release, workspace):
    _invoke(["skills", "install", "alpha"])
    skill_dir = workspace.project / ".agents" / "skills" / "alpha"
    assert skill_dir.is_dir()

    result = _invoke(["skills", "remove", "alpha", "-y"])

    assert result.exit_code == 0
    assert not skill_dir.exists()
    assert not (workspace.project / ".agents" / "skills" / ".catalog.json").exists()


def test_remove_all_removes_every_skill(release, workspace):
    _invoke(["skills", "install", "alpha", "beta"])
    skills_dir = workspace.project / ".agents" / "skills"
    assert (skills_dir / "alpha").is_dir()
    assert (skills_dir / "beta").is_dir()

    result = _invoke(["skills", "remove", "--all", "-y"])

    assert result.exit_code == 0
    assert not (skills_dir / "alpha").exists()
    assert not (skills_dir / "beta").exists()
    assert not (skills_dir / ".catalog.json").exists()


def test_remove_without_ids_errors(release, workspace):
    _invoke(["skills", "install", "alpha"])

    result = _invoke(["skills", "remove"])

    assert result.exit_code != 0
    assert "Specify skill ids to remove or pass --all" in _plain_output(result.output)


def test_remove_non_default_agent(release, workspace):
    """A skill installed to a non-default agent is removable without ``--agent``."""
    _invoke(["skills", "install", "alpha", "-a", "cursor"])
    skill_dir = workspace.project / ".cursor" / "skills" / "alpha"
    assert skill_dir.is_dir()

    result = _invoke(["skills", "remove", "alpha", "-y"])

    assert result.exit_code == 0
    assert not skill_dir.exists()


def test_fetch_failure_reports_clean_error(workspace, monkeypatch):
    """Network/parse failures surface as a clean error, not a raw traceback."""

    def boom(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr(skills_release, "fetch_release", boom)

    result = _invoke(["skills", "install", "nonexistent"])

    assert result.exit_code != 0
    assert "Could not fetch the latest skills release" in _plain_output(result.output)


def test_skills_no_subcommand_shows_help(release, workspace):
    result = _invoke(["skills"])

    assert result.exit_code == 0
    assert "install" in result.output
    assert "list" in result.output
    assert "tools" in result.output


def test_tools_help(release, workspace):
    result = _invoke(["skills", "tools", "--help"])

    assert result.exit_code == 0
    assert "Deterministic helpers" in result.output
    assert "check" in result.output
    assert "notebook" in result.output


def test_tools_forwards_subcommands(release, workspace):
    (workspace.project / "src").mkdir()

    result = _invoke(["skills", "tools", "check", "workspace", "--format", "text"])

    assert result.exit_code == 0
    assert result.output == "scaffolded\n"


async def test_auto_radio_set_selects_on_arrow(release):
    app = ProbablSkillsInstaller(agent=(), default_global=False)
    async with app.run_test() as pilot:
        await _confirm_source(app, pilot)
        app.query_one("#sel-workflows", SelectionList).select_all()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_wizard_step(app, pilot, "step-agents")
        radio = app.query_one("#agents", AutoRadioSet)
        assert radio.pressed_index == 0
        await pilot.press("down")
        await pilot.pause()
        await pilot.press("up")
        await pilot.pause()
        pressed_index = radio.pressed_index

    assert pressed_index == 0


def test_update_interactive_selection(release, workspace, monkeypatch):
    _invoke(["skills", "install", "alpha"])
    release["catalog"]["skills"][0]["hash"] = "hash-alpha-2"
    monkeypatch.setattr(_skills, "is_non_interactive", lambda: False)
    monkeypatch.setattr(
        _skills, "InstalledSkillsPicker", _fake_manage_picker(["alpha"])
    )

    result = _invoke(["skills", "update"])

    assert result.exit_code == 0
    assert "updated" in result.output


def test_update_interactive_cancelled(release, workspace, monkeypatch):
    _invoke(["skills", "install", "alpha"])
    monkeypatch.setattr(_skills, "is_non_interactive", lambda: False)
    monkeypatch.setattr(_skills, "InstalledSkillsPicker", _fake_manage_picker(None))

    result = _invoke(["skills", "update"])

    assert result.exit_code == 0
    assert "Nothing selected" in result.output


def test_remove_interactive_selection(release, workspace, monkeypatch):
    _invoke(["skills", "install", "alpha"])
    skill_dir = workspace.project / ".agents" / "skills" / "alpha"
    monkeypatch.setattr(_skills, "is_non_interactive", lambda: False)
    monkeypatch.setattr(
        _skills, "InstalledSkillsPicker", _fake_manage_picker(["alpha"])
    )

    result = _invoke(["skills", "remove", "-y"])

    assert result.exit_code == 0
    assert not skill_dir.exists()


def test_remove_interactive_cancelled(release, workspace, monkeypatch):
    _invoke(["skills", "install", "alpha"])
    skill_dir = workspace.project / ".agents" / "skills" / "alpha"
    monkeypatch.setattr(_skills, "is_non_interactive", lambda: False)
    monkeypatch.setattr(_skills, "InstalledSkillsPicker", _fake_manage_picker(None))

    result = _invoke(["skills", "remove"])

    assert result.exit_code == 0
    assert "Nothing selected" in result.output
    assert skill_dir.is_dir()


def test_remove_interactive_without_any_skill(release, workspace, monkeypatch):
    monkeypatch.setattr(_skills, "is_non_interactive", lambda: False)

    result = _invoke(["skills", "remove"])

    assert result.exit_code == 0
    assert "No skills installed" in result.output


# --------------------------------------------------------------------------- #
# Agent detection: skills install non-interactive
# --------------------------------------------------------------------------- #

_AGENT_ENV_VARS = (
    "CLAUDECODE",
    "CURSOR_AGENT",
    "CURSOR_CLI",
    "GEMINI_CLI",
    "CODEX_SANDBOX",
    "PI_CODING_AGENT",
    "OPENCODE_CLIENT",
    "CI",
)


def _clear_agent_envs(monkeypatch):
    for var in _AGENT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_install_all_keyword_installs_everything(release, workspace, monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("CLAUDECODE", "1")

    result = _invoke(["skills", "install", "all"])

    assert result.exit_code == 0
    skills_dir = workspace.project / ".claude" / "skills"
    assert (skills_dir / "alpha" / "SKILL.md").is_file()
    assert (skills_dir / "beta" / "SKILL.md").is_file()


def test_install_ids_uses_detected_agent(release, workspace, monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("CURSOR_AGENT", "1")

    result = _invoke(["skills", "install", "alpha"])

    assert result.exit_code == 0
    assert (workspace.project / ".cursor" / "skills" / "alpha").is_dir()
    assert not (workspace.project / ".agents").exists()


def test_install_non_interactive_no_args_prints_catalog(
    release, workspace, monkeypatch
):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setattr(_skills, "is_non_interactive", lambda: True)

    result = _invoke(["skills", "install"])

    assert result.exit_code == 0
    assert "alpha" in result.output
    assert "beta" in result.output
    assert "flow" in result.output
    assert "skore skills install" in result.output
    assert not (workspace.project / ".claude").exists()


def test_install_non_interactive_no_agent_prints_catalog(
    release, workspace, monkeypatch
):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setattr(_skills, "is_non_interactive", lambda: True)

    result = _invoke(["skills", "install"])

    assert result.exit_code == 0
    assert "alpha" in result.output
    assert "beta" in result.output
    assert "skore skills install" in result.output


def test_resolve_agent_names_uses_detected_agent(monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("CLAUDECODE", "1")
    assert _skills._resolve_agent_names(()) == ["claude-code"]


def test_resolve_agent_names_explicit_overrides_detection(monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("CLAUDECODE", "1")
    assert _skills._resolve_agent_names(("cursor",)) == ["cursor"]


def test_resolve_agent_names_falls_back_when_no_detection(monkeypatch):
    _clear_agent_envs(monkeypatch)
    assert _skills._resolve_agent_names(()) == ["agents"]


def test_install_custom_repo(release, workspace):
    other = {
        "skills": [
            {
                "id": "gamma",
                "path": "skills/gamma",
                "title": "Gamma",
                "summary": "The gamma skill",
                "category": "tooling",
                "hash": "hash-gamma-1",
            }
        ],
        "workflows": [],
    }
    release["by_repo"]["acme/skills"] = {"tag": "9.0.0", "catalog": other}

    result = _invoke(["skills", "install", "--repo", "acme/skills", "gamma"])

    assert result.exit_code == 0
    skill_dir = workspace.project / ".agents" / "skills" / "gamma"
    sidecar = json.loads((skill_dir / SIDECAR).read_text())
    assert sidecar == {
        "id": "gamma",
        "release": "9.0.0",
        "hash": "hash-gamma-1",
        "repository": "acme/skills",
    }
    local_catalog = json.loads(
        (workspace.project / ".agents" / "skills" / ".catalog.json").read_text()
    )
    assert set(local_catalog["sources"]) == {"acme/skills"}
    assert any("/repos/acme/skills/" in url for url in release["urls"])


def test_list_shows_custom_repo_source(release, workspace):
    other = {
        "skills": [
            {
                "id": "gamma",
                "path": "skills/gamma",
                "title": "Gamma",
                "summary": "The gamma skill",
                "category": "tooling",
                "hash": "hash-gamma-1",
            }
        ],
        "workflows": [],
    }
    release["by_repo"]["acme/skills"] = {"tag": "9.0.0", "catalog": other}
    _invoke(["skills", "install", "--repo", "acme/skills", "gamma"])

    result = _invoke(["skills", "list"])

    assert result.exit_code == 0
    assert "gamma" in result.output
    assert "acme/skills" in result.output


def test_update_uses_stored_repository(release, workspace):
    other = {
        "skills": [
            {
                "id": "gamma",
                "path": "skills/gamma",
                "title": "Gamma",
                "summary": "The gamma skill",
                "category": "tooling",
                "hash": "hash-gamma-1",
            }
        ],
        "workflows": [],
    }
    release["by_repo"]["acme/skills"] = {"tag": "9.0.0", "catalog": other}
    _invoke(["skills", "install", "--repo", "acme/skills", "gamma"])
    release["urls"].clear()
    other["skills"][0]["hash"] = "hash-gamma-2"

    result = _invoke(["skills", "update", "--all"])

    assert result.exit_code == 0
    assert "updated" in result.output
    assert "acme/skills" in result.output
    assert any("/repos/acme/skills/" in url for url in release["urls"])
    assert not any("/repos/probabl-ai/skore-skills/" in url for url in release["urls"])
    sidecar = json.loads(
        (workspace.project / ".agents" / "skills" / "gamma" / SIDECAR).read_text()
    )
    assert sidecar["hash"] == "hash-gamma-2"
    assert sidecar["repository"] == "acme/skills"


def test_update_mixed_repos_fetches_each_origin(release, workspace):
    other = {
        "skills": [
            {
                "id": "gamma",
                "path": "skills/gamma",
                "title": "Gamma",
                "summary": "The gamma skill",
                "category": "tooling",
                "hash": "hash-gamma-1",
            }
        ],
        "workflows": [],
    }
    release["by_repo"]["acme/skills"] = {"tag": "9.0.0", "catalog": other}
    _invoke(["skills", "install", "alpha"])
    _invoke(["skills", "install", "--repo", "acme/skills", "gamma"])
    release["catalog"]["skills"][0]["hash"] = "hash-alpha-2"
    other["skills"][0]["hash"] = "hash-gamma-2"
    release["urls"].clear()

    result = _invoke(["skills", "update", "--all"])

    assert result.exit_code == 0
    assert "alpha" in result.output
    assert "gamma" in result.output
    assert any("/repos/probabl-ai/skore-skills/" in url for url in release["urls"])
    assert any("/repos/acme/skills/" in url for url in release["urls"])
    local_catalog = json.loads(
        (workspace.project / ".agents" / "skills" / ".catalog.json").read_text()
    )
    assert set(local_catalog["sources"]) == {"probabl-ai/skore-skills", "acme/skills"}


def test_installed_ignores_unmanaged_entries(release, workspace):
    """Neither the catalog file nor a hand-written directory is an installed skill."""
    _invoke(["skills", "install", "alpha"])
    skills_dir = workspace.project / ".agents" / "skills"
    (skills_dir / "handwritten").mkdir()

    assert [sidecar["id"] for _, sidecar in _skills.installed(skills_dir)] == ["alpha"]


# --------------------------------------------------------------------------- #
# Multi-source remove / update edge cases
# --------------------------------------------------------------------------- #


def test_remove_keeps_catalog_source_still_in_use(release, workspace):
    """Removing one repository's skill leaves the other repository's source."""
    _register_gamma(release)
    _invoke(["skills", "install", "alpha"])
    _invoke(["skills", "install", "--repo", "acme/skills", "gamma"])
    skills_dir = workspace.project / ".agents" / "skills"

    result = _invoke(["skills", "remove", "gamma", "-y"])

    assert result.exit_code == 0
    assert not (skills_dir / "gamma").exists()
    local_catalog = json.loads((skills_dir / LOCAL_CATALOG).read_text())
    assert set(local_catalog["sources"]) == {GITHUB_REPO}


def test_remove_unknown_id_reports_nothing_to_remove(release, workspace):
    _invoke(["skills", "install", "alpha"])

    result = _invoke(["skills", "remove", "beta", "-y"])

    assert result.exit_code == 0
    assert "Nothing to remove" in result.output
    assert (workspace.project / ".agents" / "skills" / "alpha").is_dir()


def test_remove_asks_for_confirmation(release, workspace):
    _invoke(["skills", "install", "alpha"])
    skill_dir = workspace.project / ".agents" / "skills" / "alpha"

    result = _invoke(["skills", "remove", "alpha"], input="y\n")

    assert result.exit_code == 0
    assert "Removing" in result.output
    assert not skill_dir.exists()


def test_remove_declined_confirmation_keeps_skill(release, workspace):
    _invoke(["skills", "install", "alpha"])
    skill_dir = workspace.project / ".agents" / "skills" / "alpha"

    result = _invoke(["skills", "remove", "alpha"], input="n\n")

    assert result.exit_code == 0
    assert skill_dir.is_dir()


def test_update_skips_skill_absent_from_catalog(release, workspace):
    """A skill dropped from its catalog is left untouched instead of failing."""
    _invoke(["skills", "install", "alpha"])
    release["catalog"]["skills"] = [
        skill for skill in release["catalog"]["skills"] if skill["id"] != "alpha"
    ]

    result = _invoke(["skills", "update", "--all"])

    assert result.exit_code == 0
    assert "not in the latest release" in result.output
    assert "up to date" not in result.output
    assert (workspace.project / ".agents" / "skills" / "alpha" / "SKILL.md").is_file()


def test_update_unknown_id_matches_nothing(release, workspace):
    """Requesting an id that is not installed reports nothing to do."""
    _invoke(["skills", "install", "alpha"])
    release["urls"].clear()

    result = _invoke(["skills", "update", "beta"])

    assert result.exit_code == 0
    assert "All skills are up to date." in result.output
    assert release["urls"] == []


def test_install_workflow_with_unknown_skill_errors(release, workspace):
    release["catalog"]["workflows"][0]["includes"] = ["missing"]

    result = _invoke(["skills", "install", "flow"])

    assert result.exit_code != 0
    assert "references unknown skill" in result.output


# --------------------------------------------------------------------------- #
# Install wizard: GitHub source step
# --------------------------------------------------------------------------- #


async def test_wizard_app_has_no_release_before_source_confirm(release):
    app = ProbablSkillsInstaller(agent=(), default_global=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.catalog is None
        assert app.root is None
        assert app.tag is None
        assert app._selected_ids() == []
        assert app.query_one("#repo", HelpInput).value == GITHUB_REPO
        await pilot.press("escape")
        await pilot.pause()

    assert app.result is None
    assert release["urls"] == []


async def test_wizard_app_source_step_shows_help(release):
    app = ProbablSkillsInstaller(agent=(), default_global=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("?")
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert app.is_running is True


async def test_wizard_app_fetch_failure_stays_on_source(release, monkeypatch):
    def boom(repo):
        raise OSError("network down")

    monkeypatch.setattr(skills_release, "fetch_release", boom)

    app = ProbablSkillsInstaller(agent=(), default_global=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        active = app.query_one("#wizard", TabbedContent).active
        await pilot.press("escape")
        await pilot.pause()

    assert active == "step-source"
    assert app.catalog is None


async def test_wizard_app_reconfirming_same_repo_does_not_refetch(release):
    app = ProbablSkillsInstaller(agent=(), default_global=False)
    async with app.run_test() as pilot:
        await _confirm_source(app, pilot)
        release["urls"].clear()
        await _back_to_source(app, pilot)
        await _confirm_source(app, pilot)
        await pilot.press("escape")
        await pilot.pause()

    assert release["urls"] == []


async def test_wizard_app_blocks_tab_jump_before_source_confirm(release):
    app = ProbablSkillsInstaller(agent=(), default_global=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#--content-tab-step-scope")
        await pilot.pause()
        active = app.query_one("#wizard", TabbedContent).active
        await pilot.press("escape")
        await pilot.pause()

    assert active == "step-source"


async def test_wizard_app_switching_source_repo_replaces_skills(release):
    """Confirming a new repository swaps the skill lists and cleans up."""
    _register_gamma(release)

    app = ProbablSkillsInstaller(agent=("cursor",), default_global=False)
    async with app.run_test() as pilot:
        await _confirm_source(app, pilot)
        first_root = app.root
        assert set(app.query_one("#sel-skills", SelectionList).selected) == set()

        await _back_to_source(app, pilot)
        await _confirm_source(app, pilot, "acme/skills")
        skill_list = app.query_one("#sel-skills", SelectionList)
        skill_list.select_all()
        await pilot.pause()
        selected = set(skill_list.selected)
        second_root = app.root
        await pilot.press("escape")
        await pilot.pause()

    assert selected == {"gamma"}
    assert second_root != first_root
    assert not first_root.exists()
    assert app.result is None
