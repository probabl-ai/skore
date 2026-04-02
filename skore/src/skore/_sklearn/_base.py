from io import StringIO
from typing import Generic, Literal, TypeVar
from uuid import uuid4

from numpy.typing import ArrayLike
from rich.console import Console
from rich.panel import Panel

from skore._utils.repr.base import AccessorHelpMixin, ReportHelpMixin


class _BaseReport(ReportHelpMixin):
    """Base class for all reports.

    This class centralizes shared report logic (e.g. configuration, accessors) and
    inherits from ``ReportHelpMixin`` to provide a consistent ``help()`` and rich/HTML
    representation across all report types.
    """

    _ACCESSOR_CONFIG: dict[str, dict[str, str]]
    _report_type: Literal[
        "estimator",
        "cross-validation",
        "comparison-estimator",
        "comparison-cross-validation",
    ]

    def __init__(self) -> None:
        self.id = uuid4().int

    @property
    def _hash(self) -> int:
        # FIXME: only for backward compatibility
        return self.id


ParentT = TypeVar("ParentT", bound="_BaseReport")


class _BaseAccessor(AccessorHelpMixin, Generic[ParentT]):
    """Base class for all accessors.

    Accessors expose additional views on a report (e.g. data, metrics) and inherit from
    ``AccessorHelpMixin`` to provide a dedicated ``help()`` and rich/HTML help tree.
    """

    def __init__(self, parent: ParentT) -> None:
        self._parent = parent

    def _rich_repr(self, class_name: str) -> str:
        """Return a string representation using rich for accessors."""
        string_buffer = StringIO()
        console = Console(file=string_buffer, force_terminal=False)
        console.print(
            Panel(
                "Get guidance using the help() method",
                title=f"[cyan]{class_name}[/cyan]",
                border_style="orange1",
                expand=False,
            )
        )
        return string_buffer.getvalue()

    def _get_data_and_y_true(
        self,
        *,
        data_source: Literal["test", "train"],
    ) -> tuple[dict, ArrayLike]:
        """Get the requested dataset.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        Returns
        -------
        data : dict of input data
            The requested dataset.

        y : array-like of shape (n_samples,)
            The target labels.
        """
        if data_source not in ["train", "test"]:
            raise ValueError(
                f"Invalid data source: {data_source}. Possible values are: test, train."
            )
        if getattr(self._parent, f"{data_source}_data") is None:
            raise ValueError(
                f"No {data_source} data were provided when creating the report."
            )
        if data_source == "test":
            return self._parent.test_data, self._parent.y_test
        assert data_source == "train"
        return self._parent.train_data, self._parent.y_train
