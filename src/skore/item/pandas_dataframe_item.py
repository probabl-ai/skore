"""PandasDataFrameItem.

This module defines the PandasDataFrameItem class,
which represents a pandas DataFrame item.
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import cached_property

import pandas


class PandasDataFrameItem:
    """
    A class to represent a pandas DataFrame item.

    This class encapsulates a pandas DataFrame along with its
    creation and update timestamps.

    Attributes
    ----------
    dataframe_dict : Dict[str, Any]
        The dictionary representation of the pandas DataFrame.
    created_at : str
        The timestamp when the item was created, in ISO format.
    updated_at : str
        The timestamp when the item was last updated, in ISO format.

    Methods
    -------
    dataframe() : pd.DataFrame
        Returns the pandas DataFrame representation of the stored dictionary.
    factory(dataframe: pd.DataFrame) : PandasDataFrameItem
        Creates a new PandasDataFrameItem instance from a pandas DataFrame.
    """

    def __init__(
        self,
        dataframe_dict: dict,
        created_at: str,
        updated_at: str,
    ):
        self.dataframe_dict = dataframe_dict
        self.created_at = created_at
        self.updated_at = updated_at

    @cached_property
    def dataframe(self) -> pandas.DataFrame:
        """
        Convert the stored dictionary to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The pandas DataFrame representation of the stored dictionary.
        """
        return pandas.DataFrame.from_dict(self.dataframe_dict, orient="tight")

    @property
    def __dict__(self):
        """
        Get a dictionary representation of the object.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the 'dataframe_dict' key.
        """
        return {
            "dataframe_dict": self.dataframe_dict,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def factory(cls, dataframe: pandas.DataFrame) -> PandasDataFrameItem:
        """
        Create a new PandasDataFrameItem instance from a pandas DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The pandas DataFrame to store.

        Returns
        -------
        PandasDataFrameItem
            A new PandasDataFrameItem instance.
        """
        now = datetime.now(tz=UTC).isoformat()
        instance = cls(
            dataframe_dict=dataframe.to_dict(orient="tight"),
            created_at=now,
            updated_at=now,
        )

        # add dataframe as cached property
        instance.dataframe = dataframe

        return instance
