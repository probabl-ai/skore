"""Test the item notes API."""

import pytest


@pytest.mark.parametrize(
    "project_fixture",
    [
        "in_memory_project",
        "on_disk_project",
    ],
)
class TestNotes:
    def test_set_note(self, project_fixture, request):
        project = request.getfixturevalue(project_fixture)

        project.put("key", "hello")
        project.set_note("key", "note")
        assert project.get_note("key") == "note"

    def test_set_note_version(self, project_fixture, request):
        """By default, `set_note` only attaches a note to the latest version
        of a key."""
        project = request.getfixturevalue(project_fixture)

        project.put("key", "hello")
        project.put("key", "goodbye")
        project.set_note("key", "note")
        assert project.get_note("key", version=-1) == "note"
        assert project.get_note("key", version=0) is None

    def test_set_note_no_key(self, project_fixture, request):
        project = request.getfixturevalue(project_fixture)

        with pytest.raises(KeyError):
            project.set_note("key", "hello")

        project.put("key", "hello")

        with pytest.raises(KeyError):
            project.set_note("key", "hello", version=10)

    def test_set_note_not_strings(self, project_fixture, request):
        """If key or note is not a string, raise a TypeError."""
        project = request.getfixturevalue(project_fixture)

        with pytest.raises(TypeError):
            project.set_note(1, "hello")

        with pytest.raises(TypeError):
            project.set_note("key", 1)

    def test_delete_note(self, project_fixture, request):
        project = request.getfixturevalue(project_fixture)

        project.put("key", "hello")
        project.set_note("key", "note")
        project.delete_note("key")
        assert project.get_note("key") is None

    def test_delete_note_no_key(self, project_fixture, request):
        project = request.getfixturevalue(project_fixture)

        with pytest.raises(KeyError):
            project.delete_note("key")

        project.put("key", "hello")

        with pytest.raises(KeyError):
            project.set_note("key", "hello", version=10)

    def test_delete_note_no_note(self, project_fixture, request):
        project = request.getfixturevalue(project_fixture)

        project.put("key", "hello")
        assert project.get_note("key") is None

    def test_put_with_note(self, project_fixture, request):
        project = request.getfixturevalue(project_fixture)

        project.put("key", "hello", note="note")
        assert project.get_note("key") == "note"

    def test_put_with_note_annotates_latest(self, project_fixture, request):
        """Adding the `note` argument annotates the latest version of the item."""
        project = request.getfixturevalue(project_fixture)

        project.put("key", "hello")
        project.put("key", "goodbye", note="note")
        assert project.get_note("key", version=0) is None

    def test_put_with_note_wrong_type(self, project_fixture, request):
        """Adding the `note` argument annotates the latest version of the item."""
        project = request.getfixturevalue(project_fixture)

        with pytest.raises(TypeError):
            project.put("key", "hello", note=0)
