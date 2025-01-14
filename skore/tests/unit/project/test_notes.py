"""Test the item notes API."""

import pytest


def test_set_note(in_memory_project):
    in_memory_project.put("key", "hello")
    in_memory_project.set_note("key", "message")
    assert in_memory_project.get_note("key") == "message"


def test_set_note_version(in_memory_project):
    """By default, `set_note` only attaches a message to the latest version
    of a key."""
    in_memory_project.put("key", "hello")
    in_memory_project.put("key", "goodbye")
    in_memory_project.set_note("key", "message")
    assert in_memory_project.get_note("key", version=-1) == "message"
    assert in_memory_project.get_note("key", version=0) is None


def test_set_note_no_key(in_memory_project):
    with pytest.raises(KeyError):
        in_memory_project.set_note("key", "hello")

    in_memory_project.put("key", "hello")

    with pytest.raises(KeyError):
        in_memory_project.set_note("key", "hello", version=10)


def test_set_note_not_strings(in_memory_project):
    """If key or message is not a string, raise a TypeError."""
    with pytest.raises(TypeError):
        in_memory_project.set_note(1, "hello")

    with pytest.raises(TypeError):
        in_memory_project.set_note("key", 1)


def test_delete_note(in_memory_project):
    in_memory_project.put("key", "hello")
    in_memory_project.set_note("key", "message")
    in_memory_project.delete_note("key")
    assert in_memory_project.get_note("key") is None


def test_delete_note_no_key(in_memory_project):
    with pytest.raises(KeyError):
        in_memory_project.delete_note("key")

    in_memory_project.put("key", "hello")

    with pytest.raises(KeyError):
        in_memory_project.set_note("key", "hello", version=10)


def test_delete_note_no_note(in_memory_project):
    in_memory_project.put("key", "hello")
    assert in_memory_project.get_note("key") is None
