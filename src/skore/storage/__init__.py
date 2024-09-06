"""Interface and implementations of storage."""

from skore.storage.filesystem import FileSystem
from skore.storage.non_persistent_storage import NonPersistentStorage
from skore.storage.storage import Storage
from skore.storage.uri import URI

__all__ = [
    "FileSystem",
    "NonPersistentStorage",
    "Storage",
    "URI",
]
