"""
SkrubTableReportItem.

This module defines the ``SkrubTableReportItem`` class used to serialize instances of
``skrub.TableReport``, using binary protocols.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .item import ItemTypeError, lazy_is_instance, switch_mpl_backend
from .pickle_item import PickleItem

if TYPE_CHECKING:
    from skrub import TableReport


class SkrubTableReportItem(PickleItem):
    """Serialize instances of ``skrub.TableReport``, using binary protocols."""

    @property
    def __representation__(self) -> dict:
        """Get the representation of the ``SkrubTableReportItem`` instance."""
        with switch_mpl_backend():
            return {
                "representation": {
                    "media_type": "text/html",
                    "value": self.__raw__.html_snippet(),
                }
            }

    @classmethod
    def factory(cls, value: TableReport, /) -> SkrubTableReportItem:
        """
        Create a new ``SkrubTableReportItem`` from an instance of ``skrub.TableReport``.

        It uses binary protocols.

        Parameters
        ----------
        value : ``skrub.TableReport``
            The value to serialize.

        Returns
        -------
        SkrubTableReportItem
            A new ``SkrubTableReportItem`` instance.

        Raises
        ------
        ItemTypeError
            If ``value`` is not an instance of ``skrub.TableReport``.
        """
        if not lazy_is_instance(value, "skrub._reporting._table_report.TableReport"):
            raise ItemTypeError(f"Type '{value.__class__}' is not supported.")

        return super().factory(value)
