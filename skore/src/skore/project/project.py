"""Define a Project."""

from __future__ import annotations

from collections.abc import Iterator
from logging import INFO, NullHandler, getLogger
from pathlib import Path
from typing import Any, Literal, Optional, Union

from skore.persistence.item import item_to_object, object_to_item
from skore.persistence.repository import ItemRepository
from skore.persistence.storage import DiskCacheStorage

logger = getLogger(__name__)
logger.addHandler(NullHandler())  # Default to no output
logger.setLevel(INFO)


class Project:
    """
    A collection of items persisted in a storage.

    Its main methods are :func:`~skore.Project.put` and :func:`~skore.Project.get`,
    respectively to insert a key-value pair into the Project and to recover the value
    associated with a key.

    You can add any type of objects. In some cases, especially on classes you defined,
    the persistency is based on the pickle representation. You must therefore ensure
    that the call to :func:`~skore.Project.get` is made in the same environment as
    :func:`~skore.Project.put`.

    Parameters
    ----------
    path : str or Path, optional
        The path of the project to initialize, default "./project.skore".
    if_exists: Literal["raise", "load"], optional
        Raise an exception if the project already exists, or load it, default raise.

    Attributes
    ----------
    path : Path
        The unified path of the project.
    name : str
        The name of the project. Corresponds to `path.name`.

    Examples
    --------
    >>> import skore
    >>> project = skore.Project("my-xp")
    >>> project.put("score", 1.0)
    >>> project.get("score")
    1.0
    """

    _server_info = None

    def __init__(
        self,
        path: Optional[Union[str, Path]] = "project.skore",
        *,
        if_exists: Optional[Literal["raise", "load"]] = "raise",
    ):
        """
        Initialize a Project.

        Initialize a project, by creating a new project or by loading an existing one.

        Parameters
        ----------
        path : str or Path, optional
            The path of the project to initialize, default "./project.skore".
        if_exists: Literal["raise", "load"], optional
            Raise an exception if the project already exists, or load it, default raise.

        Raises
        ------
        FileExistsError
        """
        self.path = Path(path)
        self.path = self.path.with_suffix(".skore")
        self.path = self.path.resolve()
        self.name = self.path.name

        if if_exists == "raise" and self.path.exists():
            raise FileExistsError(
                f"Project '{str(path)}' already exists. Set `if_exists` to 'load' to "
                "only load the project."
            )

        item_storage_dirpath = self.path / "items"

        # Create diskcache directories
        item_storage_dirpath.mkdir(parents=True, exist_ok=True)

        # Initialize repositories with dedicated storages
        self._item_repository = ItemRepository(DiskCacheStorage(item_storage_dirpath))

        # Check if the project should rejoin a server
        from skore.project._launch import ServerInfo  # avoid circular import

        self._server_info = ServerInfo.rejoin(self)

    def clear(self):
        """Clear the project."""
        for item_key in self._item_repository:
            self._item_repository.delete_item(item_key)

    def put(
        self,
        key: str,
        value: Any,
        *,
        note: Optional[str] = None,
        display_as: Optional[Literal["HTML", "MARKDOWN", "SVG"]] = None,
    ):
        """Add a key-value pair to the Project.

        If an item with the same key already exists, its value is replaced by the new
        one.

        Parameters
        ----------
        key : str
            The key to associate with ``value`` in the Project.
        value : Any
            The value to associate with ``key`` in the Project.
        note : str, optional
            A note to attach with the item.
        display_as : {"HTML", "MARKDOWN", "SVG"}, optional
            Used in combination with a string value, it customizes the way the value is
            displayed in the interface.

        Raises
        ------
        TypeError
            If the combination of parameters are not valid.

        NotImplementedError
            If the value type is not supported.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string (found '{type(key)}')")

        self._item_repository.put_item(
            key,
            object_to_item(
                value,
                note=note,
                display_as=display_as,
            ),
        )

    def get(
        self,
        key: str,
        *,
        version: Optional[Union[Literal[-1, "all"], int]] = -1,
        metadata: bool = False,
    ):
        """Get the value associated to ``key`` from the Project.

        Parameters
        ----------
        key : str
            The key corresponding to the item to get.
        version : Union[Literal[-1, "all"], int], default=-1
            If -1, get the latest value associated to ``key``.
            If "all", get all the values associated to ``key``.
            If instance of int, get the nth value associated to ``key``.
        metadata : bool, default=False
            If True, get the metadata in addition to the value.

        Returns
        -------
        value : any
            Value associated to ``key``, when latest=True and metadata=False.
        value_and_metadata : dict
            Value associated to ``key`` with its metadata, when latest=True and
            metadata=True.
        list_of_values : list[any]
            Values associated to ``key``, ordered by date, when latest=False.
        list_of_values_and_metadata : list[dict]
            Values associated to ``key`` with their metadata, ordered by date, when
            latest=False and metadata=False.

        Raises
        ------
        KeyError
            If the key is not in the project.
        """
        if not metadata:

            def dto(item):
                return item_to_object(item)

        else:

            def dto(item):
                return {
                    "value": item_to_object(item),
                    "date": item.updated_at,
                    "note": item.note,
                }

        if version == -1:
            return dto(self._item_repository.get_item(key))
        if version == "all":
            return list(map(dto, self._item_repository.get_item_versions(key)))
        if isinstance(version, int):
            return dto(self._item_repository.get_item_versions(key)[version])

        raise ValueError('`version` should be -1, "all", or an integer')

    def keys(self) -> list[str]:
        """
        Get all keys of items stored in the project.

        Returns
        -------
        list[str]
            A list of all keys.
        """
        return self._item_repository.keys()

    def __iter__(self) -> Iterator[str]:
        """
        Yield the keys of items stored in the project.

        Returns
        -------
        Iterator[str]
            An iterator yielding all keys.
        """
        yield from self._item_repository

    def delete(self, key: str):
        """Delete the item corresponding to ``key`` from the Project.

        Parameters
        ----------
        key : str
            The key corresponding to the item to delete.

        Raises
        ------
        KeyError
            If the key does not correspond to any item.
        """
        self._item_repository.delete_item(key)

    def set_note(self, key: str, note: str, *, version=-1):
        """Attach a note to key ``key``.

        Parameters
        ----------
        key : str
            The key of the item to annotate.
            May be qualified with a version number through the ``version`` argument.
        note : str
            The note to be attached.
        version : int, default=-1
            The version of the key to annotate. Default is the latest version.

        Raises
        ------
        KeyError
            If the ``(key, version)`` couple does not exist.
        TypeError
            If ``key`` or ``note`` is not a string.

        Examples
        --------
        >>> # Annotate latest version of key "key"
        >>> project.set_note("key", "note")  # doctest: +SKIP
        >>> # Annotate first version of key "key"
        >>> project.set_note("key", "note", version=0)  # doctest: +SKIP
        """
        return self._item_repository.set_item_note(key=key, note=note, version=version)

    def get_note(self, key: str, *, version=-1) -> Union[str, None]:
        """Retrieve a note previously attached to key ``key``.

        Parameters
        ----------
        key : str
            The key of the annotated item.
            May be qualified with a version number through the ``version`` argument.
        version : int, default=-1
            The version of the annotated key. Default is the latest version.

        Returns
        -------
        The attached note, or None if no note is attached.

        Raises
        ------
        KeyError
            If the ``(key, version)`` couple does not exist.

        Examples
        --------
        >>> # Retrieve note attached to latest version of key "key"
        >>> project.get_note("key")  # doctest: +SKIP
        >>> # Retrieve note attached to first version of key "key"
        >>> project.get_note("key", version=0)  # doctest: +SKIP
        """
        return self._item_repository.get_item_note(key=key, version=version)

    def delete_note(self, key: str, *, version=-1):
        """Delete a note previously attached to key ``key``.

        If no note is attached, does nothing.

        Parameters
        ----------
        key : str
            The key of the annotated item.
            May be qualified with a version number through the ``version`` argument.
        version : int, default=-1
            The version of the annotated key. Default is the latest version.

        Raises
        ------
        KeyError
            If the ``(key, version)`` couple does not exist.

        Examples
        --------
        >>> # Delete note attached to latest version of key "key"
        >>> project.delete_note("key")  # doctest: +SKIP
        >>> # Delete note attached to first version of key "key"
        >>> project.delete_note("key", version=0)  # doctest: +SKIP
        """
        return self._item_repository.delete_item_note(key=key, version=version)

    def shutdown_web_ui(self):
        """Shutdown the web UI server if it is running."""
        if self._server_info is None:
            raise RuntimeError("UI server is not running")

        from skore.project._launch import cleanup_server  # avoid circular import

        cleanup_server(self)
