import json
import re

from click.testing import CliRunner
from textual.widgets import SelectionList, TabbedContent

from skore._cli import cli
from skore._cli.skills import _commands as _skills
from skore._cli.skills._catalog import fetch_release
from skore._cli.skills._commands import (
    ProbablSkillsFinder,
    ProbablSkillsInstaller,
)
from skore._cli.skills.app._widgets import AutoRadioSet

SIDECAR = ".skore-skill.json"
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _invoke(args):
    return CliRunner().invoke(cli, args)


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
        else:
            return
    raise AssertionError(f"wizard step {step_id!r} not ready")


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
    assert sidecar == {"id": "alpha", "release": "0.1.0", "hash": "hash-alpha-1"}


def test_install_workflow_expands_to_skills(release, workspace):
    result = _invoke(["skills", "install", "flow"])

    assert result.exit_code == 0
    skills_dir = workspace.project / ".agents" / "skills"
    assert (skills_dir / "alpha" / "SKILL.md").is_file()
    assert (skills_dir / "beta" / "SKILL.md").is_file()


def test_install_list_only_does_not_install(release, workspace):
    result = _invoke(["skills", "install", "-l"])

    assert result.exit_code == 0
    assert "flow" in result.output
    assert "alpha" in result.output
    assert not (workspace.project / ".agents").exists()


def test_install_unknown_identifier(release, workspace):
    result = _invoke(["skills", "install", "does-not-exist"])

    assert result.exit_code != 0
    assert "Unknown skill or workflow" in result.output


def test_install_explicit_agent(release, workspace):
    result = _invoke(["skills", "install", "alpha", "-a", "cursor"])

    assert result.exit_code == 0
    assert (workspace.project / ".cursor" / "skills" / "alpha").is_dir()
    assert not (workspace.project / ".agents").exists()


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


def test_install_global_without_selection_errors(release, workspace):
    result = _invoke(["skills", "install", "-g"])

    assert result.exit_code != 0
    assert not (workspace.home / ".agents").exists()


def test_install_interactive_selection(release, workspace, monkeypatch):
    monkeypatch.setattr(
        _skills,
        "_interactive_install_options",
        lambda catalog, *, agent, default_global: (
            [_skills._index(catalog)[0]["alpha"]],
            ["agents"],
            False,
        ),
    )

    result = _invoke(["skills", "install"])

    assert result.exit_code == 0
    skills_dir = workspace.project / ".agents" / "skills"
    assert (skills_dir / "alpha" / "SKILL.md").is_file()
    assert not (skills_dir / "beta").exists()


def test_install_interactive_agent_and_global(release, workspace, monkeypatch):
    monkeypatch.setattr(
        _skills,
        "_interactive_install_options",
        lambda catalog, *, agent, default_global: (
            [_skills._index(catalog)[0]["alpha"]],
            ["cursor"],
            True,
        ),
    )

    result = _invoke(["skills", "install"])

    assert result.exit_code == 0
    assert (workspace.home / ".cursor" / "skills" / "alpha").is_dir()
    assert not (workspace.project / ".cursor").exists()


def test_install_interactive_cancelled(release, workspace, monkeypatch):
    monkeypatch.setattr(
        _skills,
        "_interactive_install_options",
        lambda catalog, *, agent, default_global: None,
    )

    result = _invoke(["skills", "install"])

    assert result.exit_code == 0
    assert "Nothing selected" in result.output
    assert not (workspace.project / ".agents").exists()


def _fake_app(result):
    class _FakeApp:
        def __init__(self, catalog, *, agent, default_global):
            self.result = result

        def run(self):
            return None

    return _FakeApp


def _fake_find_app(result):
    class _FakeFindApp:
        def __init__(self, catalog):
            self.result = result

        def run(self):
            return None

    return _FakeFindApp


def _fake_menu(choice):
    class _FakeMenu:
        def __init__(self):
            self.result = choice

        def run(self):
            return None

    return _FakeMenu


def _fake_manage_picker(result):
    class _FakePicker:
        def __init__(self, skill_ids, *, title):
            self.result = result

        def run(self):
            return None

    return _FakePicker


def test_interactive_options_expands_workflow(release, monkeypatch):
    _, _, catalog = fetch_release()
    monkeypatch.setattr(
        _skills, "ProbablSkillsInstaller", _fake_app((["flow"], ["agents"], False))
    )

    selected, agents, global_ = _skills._interactive_install_options(
        catalog, agent=(), default_global=False
    )

    assert {skill["id"] for skill in selected} == {"alpha", "beta"}
    assert agents == ["agents"]
    assert global_ is False


def test_interactive_options_individual_global(release, monkeypatch):
    _, _, catalog = fetch_release()
    monkeypatch.setattr(
        _skills, "ProbablSkillsInstaller", _fake_app((["beta"], ["cursor"], True))
    )

    selected, agents, global_ = _skills._interactive_install_options(
        catalog, agent=(), default_global=False
    )

    assert {skill["id"] for skill in selected} == {"beta"}
    assert agents == ["cursor"]
    assert global_ is True


def test_interactive_options_cancelled(release, monkeypatch):
    _, _, catalog = fetch_release()
    monkeypatch.setattr(_skills, "ProbablSkillsInstaller", _fake_app(None))

    assert (
        _skills._interactive_install_options(catalog, agent=(), default_global=False)
        is None
    )


def test_interactive_options_empty_selection(release, monkeypatch):
    _, _, catalog = fetch_release()
    monkeypatch.setattr(
        _skills, "ProbablSkillsInstaller", _fake_app(([], ["agents"], False))
    )

    assert (
        _skills._interactive_install_options(catalog, agent=(), default_global=False)
        is None
    )


async def test_wizard_app_full_flow(release):
    _, _, catalog = fetch_release()

    app = ProbablSkillsInstaller(catalog, agent=(), default_global=False)
    async with app.run_test() as pilot:
        app.query_one("#sel-workflows", SelectionList).select_all()
        await pilot.pause()
        await pilot.press("enter")  # confirm skills -> agents
        await _wait_wizard_step(app, pilot, "step-agents")
        await pilot.press("enter")  # confirm agents -> scope
        await _wait_wizard_step(app, pilot, "step-scope")
        await pilot.press("enter")  # confirm scope -> install
        await pilot.pause()

    selected_ids, agents, global_ = app.result

    assert set(selected_ids) == {"flow", "alpha", "beta"}
    assert agents == ["agents"]
    assert global_ is False


async def test_wizard_app_selecting_workflow_selects_its_skills(release):
    _, _, catalog = fetch_release()

    app = ProbablSkillsInstaller(catalog, agent=(), default_global=False)
    async with app.run_test() as pilot:
        app.query_one("#sel-workflows", SelectionList).select_all()
        await _wait_workflow_skills_sync(app, pilot, {"alpha", "beta"})
        selected = list(app.query_one("#sel-skills", SelectionList).selected)

    assert set(selected) == {"alpha", "beta"}


async def test_wizard_app_deselecting_workflow_deselects_its_skills(release):
    _, _, catalog = fetch_release()

    app = ProbablSkillsInstaller(catalog, agent=(), default_global=False)
    async with app.run_test() as pilot:
        workflows = app.query_one("#sel-workflows", SelectionList)
        workflows.select_all()
        await _wait_workflow_skills_sync(app, pilot, {"alpha", "beta"})
        workflows.deselect_all()
        await _wait_workflow_skills_sync(app, pilot, set())
        selected = list(app.query_one("#sel-skills", SelectionList).selected)

    assert selected == []


async def test_wizard_app_single_agent_choice(release):
    _, _, catalog = fetch_release()

    app = ProbablSkillsInstaller(catalog, agent=(), default_global=False)
    async with app.run_test() as pilot:
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

    _, agents, _ = app.result

    assert agents == ["claude-code"]


async def test_wizard_app_skips_agent_step_when_provided(release):
    _, _, catalog = fetch_release()

    app = ProbablSkillsInstaller(catalog, agent=("cursor",), default_global=True)
    async with app.run_test() as pilot:
        app.query_one("#sel-skills", SelectionList).select_all()
        await pilot.pause()
        await pilot.press("enter")  # confirm skills -> scope (agents skipped)
        await _wait_wizard_step(app, pilot, "step-scope")
        await pilot.press("enter")  # confirm scope -> install
        await pilot.pause()

    selected_ids, agents, global_ = app.result

    assert set(selected_ids) == {"alpha", "beta"}
    assert agents == ["cursor"]
    assert global_ is True


async def test_wizard_app_cancel(release):
    _, _, catalog = fetch_release()

    app = ProbablSkillsInstaller(catalog, agent=(), default_global=False)
    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()

    assert app.result is None


async def test_wizard_app_requires_selection(release):
    _, _, catalog = fetch_release()

    app = ProbablSkillsInstaller(catalog, agent=(), default_global=False)
    async with app.run_test() as pilot:
        await pilot.press("enter")  # nothing selected -> stay on skills
        await pilot.pause()
        active = app.query_one("#wizard").active
        await pilot.press("escape")
        await pilot.pause()

    assert active == "step-skills"


def test_find_lists_all(release, workspace):
    result = _invoke(["skills", "find"])

    assert result.exit_code == 0
    assert "alpha" in result.output
    assert "beta" in result.output
    assert "flow" in result.output


def test_find_filters_by_query(release, workspace):
    result = _invoke(["skills", "find", "tooling"])

    assert result.exit_code == 0
    assert "alpha" in result.output
    assert "beta" not in result.output


def test_find_interactive_renders_selection(release, workspace, monkeypatch):
    monkeypatch.setattr(_skills, "_is_interactive", lambda: True)
    monkeypatch.setattr(_skills, "ProbablSkillsFinder", _fake_find_app(["alpha"]))

    result = _invoke(["skills", "find"])

    assert result.exit_code == 0
    assert "alpha" in result.output
    assert "beta" not in result.output


def test_find_interactive_cancelled(release, workspace, monkeypatch):
    monkeypatch.setattr(_skills, "_is_interactive", lambda: True)
    monkeypatch.setattr(_skills, "ProbablSkillsFinder", _fake_find_app(None))

    result = _invoke(["skills", "find"])

    assert result.exit_code == 0
    assert "alpha" not in result.output


async def test_finder_app_returns_selection(release):
    _, _, catalog = fetch_release()

    app = ProbablSkillsFinder(catalog)
    async with app.run_test() as pilot:
        app.query_one("#sel-workflows", SelectionList).select_all()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

    assert set(app.result) == {"flow", "alpha", "beta"}


async def test_finder_app_requires_selection(release):
    _, _, catalog = fetch_release()

    app = ProbablSkillsFinder(catalog)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
        assert app.is_running is True
        await pilot.press("escape")
        await pilot.pause()

    assert app.result is None


async def test_finder_app_cancel(release):
    _, _, catalog = fetch_release()

    app = ProbablSkillsFinder(catalog)
    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()

    assert app.result is None


def test_list_installed(release, workspace):
    _invoke(["skills", "install", "alpha"])
    result = _invoke(["skills", "list"])

    assert result.exit_code == 0
    assert "alpha" in result.output


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


def test_remove_all_removes_every_skill(release, workspace):
    _invoke(["skills", "install", "alpha", "beta"])
    skills_dir = workspace.project / ".agents" / "skills"
    assert (skills_dir / "alpha").is_dir()
    assert (skills_dir / "beta").is_dir()

    result = _invoke(["skills", "remove", "--all", "-y"])

    assert result.exit_code == 0
    assert not (skills_dir / "alpha").exists()
    assert not (skills_dir / "beta").exists()


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

    def boom():
        raise OSError("network down")

    monkeypatch.setattr(_skills, "fetch_release", boom)

    result = _invoke(["skills", "find"])

    assert result.exit_code != 0
    assert "Could not fetch the latest skills release" in _plain_output(result.output)


def test_skills_no_subcommand_shows_help(release, workspace):
    result = _invoke(["skills"])

    assert result.exit_code == 0
    assert "install" in result.output
    assert "find" in result.output


def test_skills_menu_dispatches_install(release, workspace, monkeypatch):
    monkeypatch.setattr(_skills, "_is_interactive", lambda: True)
    monkeypatch.setattr(_skills, "SkillsMenu", _fake_menu("install"))
    monkeypatch.setattr(
        _skills,
        "_interactive_install_options",
        lambda catalog, *, agent, default_global: (
            [_skills._index(catalog)[0]["alpha"]],
            ["agents"],
            False,
        ),
    )

    result = _invoke(["skills"])

    assert result.exit_code == 0
    assert (workspace.project / ".agents" / "skills" / "alpha").is_dir()


async def test_auto_radio_set_selects_on_arrow(release):
    _, _, catalog = fetch_release()

    app = ProbablSkillsInstaller(catalog, agent=(), default_global=False)
    async with app.run_test() as pilot:
        app.query_one("#sel-workflows", SelectionList).select_all()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        radio = app.query_one("#agents", AutoRadioSet)
        assert radio.pressed_index == 0
        await pilot.press("down")
        await pilot.pause()
        pressed_index = radio.pressed_index

    assert pressed_index == 1


def test_update_interactive_selection(release, workspace, monkeypatch):
    _invoke(["skills", "install", "alpha"])
    release["catalog"]["skills"][0]["hash"] = "hash-alpha-2"
    monkeypatch.setattr(_skills, "_is_interactive", lambda: True)
    monkeypatch.setattr(
        _skills, "InstalledSkillsPicker", _fake_manage_picker(["alpha"])
    )

    result = _invoke(["skills", "update"])

    assert result.exit_code == 0
    assert "updated" in result.output


def test_update_interactive_cancelled(release, workspace, monkeypatch):
    _invoke(["skills", "install", "alpha"])
    monkeypatch.setattr(_skills, "_is_interactive", lambda: True)
    monkeypatch.setattr(_skills, "InstalledSkillsPicker", _fake_manage_picker(None))

    result = _invoke(["skills", "update"])

    assert result.exit_code == 0
    assert "Nothing selected" in result.output


def test_remove_interactive_selection(release, workspace, monkeypatch):
    _invoke(["skills", "install", "alpha"])
    skill_dir = workspace.project / ".agents" / "skills" / "alpha"
    monkeypatch.setattr(_skills, "_is_interactive", lambda: True)
    monkeypatch.setattr(
        _skills, "InstalledSkillsPicker", _fake_manage_picker(["alpha"])
    )

    result = _invoke(["skills", "remove", "-y"])

    assert result.exit_code == 0
    assert not skill_dir.exists()


def test_remove_interactive_cancelled(release, workspace, monkeypatch):
    _invoke(["skills", "install", "alpha"])
    skill_dir = workspace.project / ".agents" / "skills" / "alpha"
    monkeypatch.setattr(_skills, "_is_interactive", lambda: True)
    monkeypatch.setattr(_skills, "InstalledSkillsPicker", _fake_manage_picker(None))

    result = _invoke(["skills", "remove"])

    assert result.exit_code == 0
    assert "Nothing selected" in result.output
    assert skill_dir.is_dir()
