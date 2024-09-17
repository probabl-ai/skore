"""PandasDataFrameItem.

This module defines the PandasDataFrameItem class,
which represents a pandas DataFrame item.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas

from skore.item.item import Item


class PandasDataFrameItem(Item):
    """
    A class to represent a pandas DataFrame item.

    This class encapsulates a pandas DataFrame along with its
    creation and update timestamps.
    """

    def __init__(
        self,
        dataframe_dict: dict,
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        """
        Initialize a PandasDataFrameItem.

        Parameters
        ----------
        dataframe_dict : dict
            The dict representation of the dataframe.
        created_at : str
            The creation timestamp in ISO format.
        updated_at : str
            The last update timestamp in ISO format.
        """
        super().__init__(created_at, updated_at)

        self.dataframe_dict = dataframe_dict

    @cached_property
    def dataframe(self) -> pandas.DataFrame:
        """
        Convert the stored dictionary to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The pandas DataFrame representation of the stored dictionary.
        """
        import pandas

        return pandas.DataFrame.from_dict(self.dataframe_dict, orient="tight")

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
        instance = cls(dataframe_dict=dataframe.to_dict(orient="tight"))

        # add dataframe as cached property
        instance.dataframe = dataframe

        return instance
