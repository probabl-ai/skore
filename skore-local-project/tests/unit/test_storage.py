import os
import shutil

import pytest

from skore_local_project.storage import DiskCacheStorage


class TestDiskCacheStorage:
    @pytest.fixture
    def storage(self, tmp_path):
        return DiskCacheStorage(tmp_path)

    def test_contains(self, storage):
        storage["<key1>"] = "<value1>"
        storage["<key2>"] = "<value2>"

        assert "<key1>" in storage
        assert "<key2>" in storage
        assert "<key3>" not in storage

    def test_len(self, storage):
        assert len(storage) == 0

        storage["<key1>"] = "<value1>"
        storage["<key2>"] = "<value2>"

        assert len(storage) == 2

    def test_iter(self, storage):
        storage["<key1>"] = "<value1>"
        storage["<key2>"] = "<value2>"

        assert list(storage) == ["<key1>", "<key2>"]

    def test_getitem(self, storage):
        storage["<key1>"] = "<value1>"
        storage["<key2>"] = "<value2>"

        assert storage["<key1>"] == "<value1>"
        assert storage["<key2>"] == "<value2>"

    def test_delitem(self, storage):
        assert len(storage) == 0

        storage["<key1>"] = "<value1>"
        storage["<key2>"] = "<value2>"

        assert len(storage) == 2

        del storage["<key2>"]

        assert len(storage) == 1

        del storage["<key1>"]

        assert len(storage) == 0

    def test_keys(self, storage):
        storage["<key2>"] = "<value2>"
        storage["<key1>"] = "<value1>"

        assert list(storage.keys()) == ["<key2>", "<key1>"]

    def test_values(self, storage):
        storage["<key2>"] = "<value2>"
        storage["<key1>"] = "<value1>"

        assert list(storage.values()) == ["<value2>", "<value1>"]

    def test_items(self, storage):
        storage["<key1>"] = "<value1>"
        storage["<key2>"] = "<value2>"

        assert list(storage.items()) == [("<key1>", "<value1>"), ("<key2>", "<value2>")]

    def test_autoreload(self, tmp_path):
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
