"""PandasDataFrameItem.

This module defines the PandasDataFrameItem class,
which represents a pandas DataFrame item.
"""

from __future__ import annotations

import io
from functools import cached_property
from typing import TYPE_CHECKING, Union

import pyarrow
import pyarrow.parquet

from skore.item.item import Item, ItemTypeError

if TYPE_CHECKING:
    import pandas


class PandasDataFrameItem(Item):
    """
    A class to represent a pandas DataFrame item.

    This class encapsulates a pandas DataFrame along with its
    creation and update timestamps.
    """

    def __init__(
        self,
        dataframe_bytes: bytes,
        created_at: Union[str, None] = None,
        updated_at: Union[str, None] = None,
        note: Union[str, None] = None,
    ):
        """
        Initialize a PandasDataFrameItem.

        Parameters
        ----------
        dataframe_bytes : str
            The dataframe, as bytes.
        created_at : str
            The creation timestamp in ISO format.
        updated_at : str
            The last update timestamp in ISO format.
        note : Union[str, None]
            An optional note.
        """
        super().__init__(created_at, updated_at, note)

        self.dataframe_bytes = dataframe_bytes

    @cached_property
    def dataframe(self) -> pandas.DataFrame:
        """
        The pandas DataFrame from the persistence.

        Its content can differ from the original dataframe, in order to be
        environment-independent.
        """
        with io.BytesIO(self.dataframe_bytes) as df_stream:
            table = pyarrow.parquet.read_table(df_stream)
            dataframe = table.to_pandas()

            return dataframe

    def as_serializable_dict(self):
        """Get a serializable dict from the item.

        Derived class must call their super implementation
        and merge the result with their output.
        """
        d = super().as_serializable_dict()
        d.update(
            {
                "media_type": "application/vnd.dataframe",
                "value": self.dataframe.fillna("NaN").to_dict(orient="tight"),
            }
        )
        return d

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
        import pandas

        if not isinstance(dataframe, pandas.DataFrame):
            raise ItemTypeError(f"Type '{dataframe.__class__}' is not supported.")

        table = pyarrow.Table.from_pandas(dataframe)

        with io.BytesIO() as dataframe_bytes:
            pyarrow.parquet.write_table(table, dataframe_bytes)

            return cls(dataframe_bytes=dataframe_bytes.getvalue())
