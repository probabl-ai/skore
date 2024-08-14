from pathlib import Path

import pytest
from mandr.storage import FileSystem
from mandr.store import Store
from mandr.store.store import MandrRootException


class TestDefaultStorage:
    """Test the behaviour of initialising a Mandr when `storage` is not given."""

    def test_absolute_path(self, monkeypatch, tmp_path):
        """If MANDR_ROOT is an absolute path, the storage is in MANDR_ROOT."""
        monkeypatch.setenv("MANDR_ROOT", str(tmp_path))

        store = Store("root/probabl")

        assert isinstance(store.storage, FileSystem)
        assert Path(store.storage.cache.directory) == tmp_path

    def test_relative_path(self, monkeypatch, tmp_path):
        """If MANDR_ROOT is a relative path, we raise an error."""
        monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
        monkeypatch.setenv("MANDR_ROOT", ".not_datamander")

        with pytest.raises(MandrRootException, match="got '.not_datamander'"):
            Store("root/probabl")

    def test_relative_path_no_mandr_root(self, monkeypatch, tmp_path):
        """If MANDR_ROOT is unset, the storage is in ".datamander",
        relative to the current working directory."""
        monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)

        store = Store("root/probabl")

        assert isinstance(store.storage, FileSystem)
        assert Path(store.storage.cache.directory) == tmp_path / ".datamander"
