"""CrossValidateItem class.

This class represents the output of a cross-validation workflow.
"""

from __future__ import annotations

from skore.item.item import Item


class CrossValidateItem(Item):
    """
    A class to represent the output of a cross-validation workflow.

    This class encapsulates the output of scikit-learn's cross-validate function along
    with its creation and update timestamps.
    """

    def __init__(
        self,
        cv_results: dict,
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        """
        Initialize a CrossValidateItem.

        Parameters
        ----------
        cv_results: dict
            The dict output of scikit-learn's cross_validate function.
        created_at : str
            The creation timestamp in ISO format.
        updated_at : str
            The last update timestamp in ISO format.
        """
        super().__init__(created_at, updated_at)

        self.cv_results = cv_results

    @classmethod
    def factory(cls, cv_results: dict) -> CrossValidateItem:
        """
        Create a new CrossValidateItem instance.

        Parameters
        ----------
        cv_results : dict
            The dict output of scikit-learn's cross_validate function.

        Returns
        -------
        CrossValidateItem
            A new CrossValidateItem instance.
        """
        if not isinstance(cv_results, dict):
            raise TypeError(f"Type '{cv_results.__class__}' is not supported.")

        instance = cls(cv_results=cv_results)

        return instance
