"""Define a Project."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from skore.persistence import item_to_object, object_to_item

if TYPE_CHECKING:
    from skore.persistence import (
        Item,
        ItemRepository,
        View,
        ViewRepository,
    )


MISSING = object()


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

    def put(self, key: Union[str, dict[str, Any]], value: Optional[Any] = MISSING):
        """Add one or more key-value pairs to the Project.

        If an item with the same key already exists, its value is replaced by the new
        one.

        If ``key`` is a string, then :func:`~skore.Project.put` adds the single
        ``key``-``value`` pair mapping to the Project.
        If ``key`` is a dict, it is interpreted as multiple key-value pairs to add to
        the Project.

        The dict format is equivalent to running :func:`~skore.Project.put`
        for each individual key-value pair. In other words,

        .. code-block:: python

            project.put({"hello": 1, "goodbye": 2})

        is equivalent to

        .. code-block:: python

            project.put("hello", 1)
            project.put("goodbye", 2)

        In particular, this means that if some key-value pair is invalid
        (e.g. if a key is not a string, or a value's type is not supported),
        then all the key-value pairs up to the first offending key-value pair will
        be successfully inserted, *and then* an error will be raised.

        Parameters
        ----------
        key : str | dict[str, Any]
            The key to associate with ``value`` in the Project,
            or dict of key-value pairs to add to the Project.
        value : Any, optional
            The value to associate with ``key`` in the Project.
            If ``key`` is a dict, this argument is ignored.

        Raises
        ------
        TypeError
            If the combination of parameters are not valid.

        NotImplementedError
            If the value type is not supported.
        """
        if value is not MISSING:
            key_to_value = {key: value}
        elif isinstance(key, dict):
            key_to_value = key
        else:
            raise TypeError(
                f"Bad parameters. "
                f"When value is not specified, key must be a dict (found '{type(key)}')"
            )

        for key, value in key_to_value.items():
            if not isinstance(key, str):
                raise TypeError(f"Key must be a string (found '{type(key)}')")

            self.item_repository.put_item(key, object_to_item(value))

    def get(self, key: str) -> Any:
        """Get the value corresponding to ``key`` from the Project.

        Parameters
        ----------
        key : str
            The key corresponding to the item to get.

        Raises
        ------
        KeyError
            If the key does not correspond to any item.
        """
        return item_to_object(self.item_repository.get_item(key))

    def get_item_versions(self, key: str) -> list[Item]:
        """
        Get all the versions of an item associated with ``key`` from the Project.

        The list is ordered from oldest to newest :func:`~skore.Project.put` date.

        Parameters
        ----------
        key : str
            The key corresponding to the item to get.

        Returns
        -------
        list[Item]
            The list of items corresponding to ``key``.

        Raises
        ------
        KeyError
            If the key does not correspond to any item.
        """
        return self.item_repository.get_item_versions(key)

    def list_item_keys(self) -> list[str]:
        """List all item keys in the Project.

        Returns
        -------
        list[str]
            The list of item keys. The list is empty if there is no item.
        """
        return self.item_repository.keys()

    def delete_item(self, key: str):
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

    def put_view(self, key: str, view: View):
        """Add a view to the Project."""
        self.view_repository.put_view(key, view)

    def get_view(self, key: str) -> View:
        """Get the view corresponding to ``key`` from the Project.

        Parameters
        ----------
        key : str
            The key of the item to get.

        Returns
        -------
        View
            The view corresponding to ``key``.

        Raises
        ------
        KeyError
            If the key does not correspond to any view.
        """
        return self.view_repository.get_view(key)

    def delete_view(self, key: str):
        """Delete the view corresponding to ``key`` from the Project.

        Parameters
        ----------
        key : str
            The key corresponding to the view to delete.

        Raises
        ------
        KeyError
            If the key does not correspond to any view.
        """
        return self.view_repository.delete_view(key)

    def list_view_keys(self) -> list[str]:
        """List all view keys in the Project.

        Returns
        -------
        list[str]
            The list of view keys. The list is empty if there is no view.
        """
        return self.view_repository.keys()
