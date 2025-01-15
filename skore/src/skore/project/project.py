"""Define a Project."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from skore.persistence.item import item_to_object, object_to_item

if TYPE_CHECKING:
    from skore.persistence import (
        ItemRepository,
        ViewRepository,
    )


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Default to no output
logger.setLevel(logging.INFO)


class Project:
    """
    A collection of items arranged in views and stored in a storage.

    Its main methods are :func:`~skore.Project.put` and :func:`~skore.Project.get`,
    respectively to insert a key-value pair into the Project and to recover the value
    associated with a key.

    There is no guarantee of getting back the exact same object that was put in,
    because ``skore`` strives to save information independently of the environment
    (so that it can be recovered in spite of version changes).
    Similarly, ``put`` raises an exception if the type of the inserted value is not
    supported.
    For example, this is the case for custom-made classes:

    .. code-block:: python

        class A:
            pass

        p.put('hello', A())
        # NotImplementedError: Type '<class '__main__.A'>' is not supported.

    However, when possible, ``get`` will return an object as close as possible to
    what was ``put`` in. Here is a summary of what types of data are supported and to
    what extent:

    * JSON-serializable ("primitive") values, like Python ints, floats, and strings,
      as well as tuples, lists and dicts of primitive values, are fully supported:

        .. code-block:: python

            project.put("my-key", {1: 'a', 'b': ('2', [3.5, 4])})
            project.get("my-key")
            # {1: 'a', 'b': ('2', [3.5, 4])}

    * numpy arrays, pandas DataFrames and Series, and polars DataFrames and Series are
      fully supported.

    * matplotlib Figures, plotly Figures, pillow Images and altair Charts are supported
      by ``put``, but ``get`` will only recover the raw data.
      However some of these libraries support importing this raw data.
    """

    def __init__(
        self,
        item_repository: ItemRepository,
        view_repository: ViewRepository,
    ):
        self.item_repository = item_repository
        self.view_repository = view_repository

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
        display_as : str, optional
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

        self.item_repository.put_item(
            key,
            object_to_item(
                value,
                note=note,
                display_as=display_as,
            ),
        )

    def get(self, key, *, latest=True, metadata=False):
        """Get the value associated to ``key`` from the Project.

        Parameters
        ----------
        key : str
            The key corresponding to the item to get.
        latest : boolean, optional
            Get the latest value or all the values associated to ``key``, default True.
        metadata : boolean, optional
            Get the metadata in addition of the value, default False.

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

        if latest:
            return dto(self.item_repository.get_item(key))
        return list(map(dto, self.item_repository.get_item_versions(key)))

    def keys(self) -> list[str]:
        """
        Get all keys of items stored in the project.

        Returns
        -------
        list[str]
            A list of all keys.
        """
        return self.item_repository.keys()

    def __iter__(self) -> Iterator[str]:
        """
        Yield the keys of items stored in the project.

        Returns
        -------
        Iterator[str]
            An iterator yielding all keys.
        """
        yield from self.item_repository

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
        self.item_repository.delete_item(key)

    def set_note(self, key: str, message: str, *, version=-1):
        """Attach a note to key ``key``.

        Parameters
        ----------
        key : str
            The key of the item to annotate.
            May be qualified with a version number through the ``version`` argument.
        message : str
            The message to be attached.
        version : int, default=-1
            The version of the key to annotate. Default is the latest version.

        Raises
        ------
        KeyError
            If the ``(key, version)`` couple does not exist.
        TypeError
            If ``key`` or ``message`` is not a string.

        Examples
        --------
        # Annotate latest version of key "key"
        >>> project.set_note("key", "message")  # doctest: +SKIP

        # Annotate first version of key "key"
        >>> project.set_note("key", "message", version=0)  # doctest: +SKIP
        """
        return self.item_repository.set_item_note(
            key=key, message=message, version=version
        )

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
        # Retrieve note attached to latest version of key "key"
        >>> project.get_note("key")  # doctest: +SKIP

        # Retrieve note attached to first version of key "key"
        >>> project.get_note("key", version=0)  # doctest: +SKIP
        """
        return self.item_repository.get_item_note(key=key, version=version)

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
        # Delete note attached to latest version of key "key"
        >>> project.delete_note("key")  # doctest: +SKIP

        # Delete note attached to first version of key "key"
        >>> project.delete_note("key", version=0)  # doctest: +SKIP
        """
        return self.item_repository.delete_item_note(key=key, version=version)
