"""Implement a repository for Reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skore.persistence.abstract_storage import AbstractStorage
    from skore.report.report import Report


class ReportRepository:
    """
    A repository for managing storage and retrieval of Reports.

    This class provides methods to get, put, and delete Reports from a storage system.
    """

    def __init__(self, storage: AbstractStorage):
        """
        Initialize the ReportRepository with a storage system.

        Parameters
        ----------
        storage : AbstractStorage
            The storage system to be used by the repository.
        """
        self.storage = storage

    def get_report(self, key: str) -> Report:
        """
        Retrieve the Report from storage.

        Parameters
        ----------
        key : str
            A key at which to look for a Report.

        Returns
        -------
        Report
            The retrieved Report.

        Raises
        ------
        KeyError
            When `key` is not present in the underlying storage.
        """
        return self.storage[key]

    def put_report(self, key: str, report: Report):
        """
        Store a report in storage.

        Parameters
        ----------
        report : Report
            The report to be stored.
        """
        self.storage[key] = report

    def delete_report(self, key: str):
        """Delete the report from storage."""
        del self.storage[key]

    def keys(self) -> list[str]:
        """
        Get all keys of items stored in the repository.

        Returns
        -------
        list[str]
            A list of all keys in the storage.
        """
        return list(self.storage.keys())
