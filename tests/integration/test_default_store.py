from pathlib import Path

from skore.storage import FileSystem
from skore.store import Store


class TestDefaultStorage:
    """Test the behaviour of initialising a Skore when `storage` is not given."""

    def test_absolute_path(self, monkeypatch, tmp_path):
        """If SKORE_ROOT is an absolute path, the storage is in SKORE_ROOT."""
        monkeypatch.setenv("SKORE_ROOT", str(tmp_path))

        store = Store("root/probabl")

        assert isinstance(store.storage, FileSystem)
        assert Path(store.storage.cache.directory) == tmp_path

    def test_relative_path(self, monkeypatch, tmp_path):
        """If SKORE_ROOT is a relative path, we raise an error."""
        monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
        monkeypatch.setenv("SKORE_ROOT", ".datamander")

        store = Store("root/probabl")

        assert isinstance(store.storage, FileSystem)
        assert Path(store.storage.cache.directory) == tmp_path / ".datamander"

    def test_relative_path_no_SKORE_ROOT(self, monkeypatch, tmp_path):
        """If SKORE_ROOT is unset, the storage is in ".datamander",
        relative to the current working directory."""
        monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)

        store = Store("root/probabl")

        assert isinstance(store.storage, FileSystem)
        assert Path(store.storage.cache.directory) == tmp_path / ".datamander"
