"""Define a Project."""

import logging
from typing import Any, Optional, Union

from skore.item import (
    CrossValidationItem,
    Item,
    ItemRepository,
    MediaItem,
    NumpyArrayItem,
    PandasDataFrameItem,
    PandasSeriesItem,
    PolarsDataFrameItem,
    PolarsSeriesItem,
    PrimitiveItem,
    SklearnBaseEstimatorItem,
    object_to_item,
)
from skore.view.view import View
from skore.view.view_repository import ViewRepository

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

    _server_manager = None

    def __init__(
        self,
        name: str,
        item_repository: ItemRepository,
        view_repository: ViewRepository,
    ):
        self.name = name
        self.item_repository = item_repository
        self.view_repository = view_repository

    def put(self, key: str, value: Any, *, note: Optional[str] = None):
        """Add a key-value pair to the Project.

        If an item with the same key already exists, its value is replaced by the new
        one.

        Parameters
        ----------
        key : str
            The key to associate with ``value`` in the Project.
        value : Any
            The value to associate with ``key`` in the Project.
        note : str or None, optional
            A note to attach with the item.

        Raises
        ------
        TypeError
            If the combination of parameters are not valid.

        NotImplementedError
            If the value type is not supported.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string (found '{type(key)}')")

        item = object_to_item(value)

        if note is not None:
            if not isinstance(note, str):
                raise TypeError(f"Note must be a string (found '{type(note)}')")
            item.note = note

        self.item_repository.put_item(key, item)

    def put_item(self, key: str, item: Item):
        """Add an Item to the Project."""
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string (found '{type(key)}')")

        self.item_repository.put_item(key, item)

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
        item = self.get_item(key)

        if isinstance(item, PrimitiveItem):
            return item.primitive
        elif isinstance(item, NumpyArrayItem):
            return item.array
        elif isinstance(item, PandasDataFrameItem):
            return item.dataframe
        elif isinstance(item, PandasSeriesItem):
            return item.series
        elif isinstance(item, PolarsDataFrameItem):
            return item.dataframe
        elif isinstance(item, PolarsSeriesItem):
            return item.series
        elif isinstance(item, SklearnBaseEstimatorItem):
            return item.estimator
        elif isinstance(item, CrossValidationItem):
            return item.cv_results_serialized
        elif isinstance(item, MediaItem):
            return item.media_bytes
        else:
            raise ValueError(f"Item {item} is not a known item type.")

    def get_item(self, key: str) -> Item:
        """Get the item corresponding to ``key`` from the Project.

        Parameters
        ----------
        key : str
            The key corresponding to the item to get.

        Returns
        -------
        item : Item
            The Item corresponding to ``key``.

        Raises
        ------
        KeyError
            If the key does not correspond to any item.
        """
        return self.item_repository.get_item(key)

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
        # Annotate latest version of key "key"
        >>> project.set_note("key", "note")  # doctest: +SKIP

        # Annotate first version of key "key"
        >>> project.set_note("key", "note", version=0)  # doctest: +SKIP
        """
        return self.item_repository.set_item_note(key=key, note=note, version=version)

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

    def shutdown_web_ui(self):
        """Shutdown the web UI server if it is running."""
        if self._server_manager is not None:
            self._server_manager.shutdown()
        else:
            raise RuntimeError("UI server is not running")
