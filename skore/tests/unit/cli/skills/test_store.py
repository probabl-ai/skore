"""Local catalog files that are not valid JSON."""

from __future__ import annotations

from pathlib import Path

import pytest

from skore._cli.skills import release as skills_release
from skore._cli.skills.release import open_release
from skore._cli.skills.store import (
    SkillStoreError,
    install_skill,
    installed,
    read_local_catalog,
)


def test_read_local_catalog_rejects_malformed_json(tmp_path):
    catalog = tmp_path / ".catalog.json"
    catalog.write_text("{not json\n")

    with pytest.raises(SkillStoreError, match="could not parse"):
        read_local_catalog(tmp_path)


def test_read_local_catalog_rejects_non_object(tmp_path):
    (tmp_path / ".catalog.json").write_text("[]\n")

    with pytest.raises(SkillStoreError, match="JSON object"):
        read_local_catalog(tmp_path)


def test_installed_rejects_sidecar_without_id(tmp_path):
    skill = tmp_path / "alpha"
    skill.mkdir()
    (skill / ".skore-skill.json").write_text("[]\n")

    with pytest.raises(SkillStoreError, match="object with an id"):
        list(installed(tmp_path))


@pytest.mark.parametrize(
    "skill_id",
    ["../../.ssh/authorized_keys", "..", "/tmp/skill", "foo/bar", ""],
)
def test_install_skill_rejects_an_unsafe_id(tmp_path, skill_id):
    root = tmp_path / "release"
    root.mkdir()

    with pytest.raises(SkillStoreError, match="skill id"):
        install_skill(
            {"id": skill_id, "path": "skills/alpha", "hash": "h"},
            root,
            tmp_path / "skills",
            "v1",
            "owner/name",
        )

    assert not (tmp_path / "skills").exists()
    assert list(root.iterdir()) == []


def test_install_skill_rejects_an_absolute_path(tmp_path):
    root = tmp_path / "release"
    root.mkdir()

    with pytest.raises(SkillStoreError, match="escapes the release"):
        install_skill(
            {"id": "alpha", "path": "/etc/passwd", "hash": "h"},
            root,
            tmp_path / "skills",
            "v1",
            "owner/name",
        )


def test_install_skill_rejects_a_path_outside_the_release(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text("secret")

    with pytest.raises(SkillStoreError, match="escapes the release"):
        install_skill(
            {"id": "alpha", "path": "../outside", "hash": "h"},
            root,
            tmp_path / "skills",
            "v1",
            "owner/name",
        )

    assert not (tmp_path / "skills").exists()


def test_install_skill_keeps_existing_directory_when_copy_fails(tmp_path):
    target = tmp_path / "skills"
    destination = target / "alpha"
    destination.mkdir(parents=True)
    (destination / "SKILL.md").write_text("old")
    root = tmp_path / "release"
    root.mkdir()

    with pytest.raises(FileNotFoundError):
        install_skill(
            {"id": "alpha", "path": "missing", "hash": "h"},
            root,
            target,
            "v1",
            "owner/name",
        )

    assert (destination / "SKILL.md").read_text() == "old"


def test_install_skill_replaces_leftovers_and_restores_on_rename_failure(
    tmp_path, monkeypatch
):
    root = tmp_path / "release"
    source = root / "skills" / "alpha"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text("new")
    target = tmp_path / "skills"
    destination = target / "alpha"
    destination.mkdir(parents=True)
    (destination / "SKILL.md").write_text("old")
    (target / ".alpha.skore-tmp").mkdir()
    (target / ".alpha.skore-bak").mkdir()
    original_rename = Path.rename

    def rename(self, dst):
        if self.name.endswith(".skore-tmp"):
            raise OSError("rename failed")
        return original_rename(self, dst)

    monkeypatch.setattr(Path, "rename", rename)

    with pytest.raises(OSError, match="rename failed"):
        install_skill(
            {"id": "alpha", "path": "skills/alpha", "hash": "h"},
            root,
            target,
            "v1",
            "owner/name",
        )

    assert (destination / "SKILL.md").read_text() == "old"


def test_install_skill_propagates_rename_failure_for_a_new_skill(tmp_path, monkeypatch):
    root = tmp_path / "release"
    source = root / "skills" / "alpha"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text("new")
    target = tmp_path / "skills"
    target.mkdir()
    original_rename = Path.rename

    def rename(self, dst):
        if self.name.endswith(".skore-tmp"):
            raise OSError("rename failed")
        return original_rename(self, dst)

    monkeypatch.setattr(Path, "rename", rename)

    with pytest.raises(OSError, match="rename failed"):
        install_skill(
            {"id": "alpha", "path": "skills/alpha", "hash": "h"},
            root,
            target,
            "v1",
            "owner/name",
        )

    assert not (target / "alpha").exists()


def test_open_release_removes_the_extraction_directory(tmp_path, monkeypatch):
    root = tmp_path / "extract" / "repo"
    root.mkdir(parents=True)

    monkeypatch.setattr(
        skills_release,
        "load_release",
        lambda repo: ("v1", root, {"skills": []}),
    )

    with open_release("owner/name") as (tag, yielded, catalog):
        assert tag == "v1"
        assert yielded == root
        assert catalog == {"skills": []}

    assert not root.exists()
