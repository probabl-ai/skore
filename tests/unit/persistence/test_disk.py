import os
import shutil
from pathlib import Path

from skore.persistence.storage import DiskCacheStorage


def test_disk_storage(tmp_path: Path):
    storage = DiskCacheStorage(tmp_path)
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

    assert repr(storage) == f"DiskCacheStorage(directory='{tmp_path}')"


def test_autoreload(tmp_path: Path):
    os.mkdir(tmp_path / "test1/")
    storage1 = DiskCacheStorage(tmp_path / "test1/")
    storage1["key"] = "test1"

    assert storage1["key"] == "test1"

    os.mkdir(tmp_path / "test2/")
    storage2 = DiskCacheStorage(tmp_path / "test2/")
    storage2["key"] = "test2"

    assert storage2["key"] == "test2"

    shutil.rmtree(tmp_path / "test1/")
    shutil.copytree(tmp_path / "test2/", tmp_path / "test1/")

    assert storage1["key"] == "test2"
    assert storage2["key"] == "test2"
