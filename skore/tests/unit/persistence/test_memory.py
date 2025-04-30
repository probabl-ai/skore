from skore.persistence.storage import InMemoryStorage


def test_in_memory_storage():
    storage = InMemoryStorage()
    storage["key"] = "value"

    assert storage["key"] == "value"
    assert "key" in storage
    assert len(storage) == 1
    assert list(storage.keys()) == ["key"]
    assert list(storage.values()) == ["value"]
    assert list(storage.items()) == [("key", "value")]

    del storage["key"]
    assert "key" not in storage
    assert len(storage) == 0
    assert list(storage.keys()) == []
    assert list(storage.values()) == []
    assert list(storage.items()) == []

    assert repr(storage) == "InMemoryStorage()"
