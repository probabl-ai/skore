"""Unit tests for ``skore._utils.repr.rich_repr``."""

from io import StringIO

import numpy as np
import pytest
from rich.console import Console
from sklearn.linear_model import LogisticRegression

from skore._utils._testing import MockAccessor, MockDisplay, MockReport
from skore._utils.repr.rich_repr import (
    _RichAccessorHelpMixin,
    _RichHelpDisplayMixin,
    _RichReportHelpMixin,
)


def _render_panel(panel):
    """Render a Rich Panel to a plain string."""
    buf = StringIO()
    Console(file=buf, force_terminal=False).print(panel)
    return buf.getvalue()


class _AccessorWithRich(MockAccessor, _RichAccessorHelpMixin):
    """Minimal accessor with Rich help mixin for tests."""

    _ACCESSOR_CONFIG: dict = {}

    def _get_help_title(self) -> str:
        return "Mock accessor"

    def fetch(self):
        """Fetch data (enables method entries in tree)."""
        pass


class _ReportWithRich(MockReport, _RichReportHelpMixin):
    """Minimal report with Rich help mixin for tests."""

    def public_action(self):
        """A public method (enables Methods section)."""
        pass


class _ReportWithAccessorAndRich(MockReport, _RichReportHelpMixin):
    """Report with _ACCESSOR_CONFIG and metrics accessor (enables accessors loop)."""

    _ACCESSOR_CONFIG = {"metrics": {"name": "metrics"}}

    def __init__(self, estimator, X_train=None, y_train=None, X_test=None, y_test=None):
        super().__init__(
            estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )
        self.metrics = _AccessorWithRich(parent=self)

    def public_action(self):
        """A public method (enables Methods section)."""
        pass


class _DisplayWithRich(MockDisplay, _RichHelpDisplayMixin):
    """Minimal display with Rich help mixin for tests."""

    _ACCESSOR_CONFIG: dict = {}
    some_attr = "value"

    def _get_help_title(self) -> str:
        return "Mock display"


@pytest.fixture
def report_with_rich():
    """Report with Rich help mixin and metrics accessor (covers accessors loop)."""
    X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y = np.array([0, 1, 1, 0])
    estimator = LogisticRegression().fit(X, y)
    return _ReportWithAccessorAndRich(estimator)


@pytest.fixture
def accessor_with_rich(report_with_rich):
    """Accessor with Rich help mixin."""
    return _AccessorWithRich(parent=report_with_rich)


@pytest.fixture
def display_with_rich():
    """Display with Rich help mixin."""
    return _DisplayWithRich()


def test_rich_report_help_mixin_tree_and_title(report_with_rich):
    """_RichReportHelpMixin tree contains accessors, Methods/Attributes and panel
    title."""
    panel = report_with_rich._create_help_panel()
    out = _render_panel(panel)
    for expected in (
        "Mock report",
        "metrics",
        "fetch",
        "Methods",
        "Attributes",
        report_with_rich.__class__.__name__,
    ):
        assert expected in out


def test_rich_accessor_help_mixin_tree_and_title(accessor_with_rich):
    """_RichAccessorHelpMixin tree contains root node, accessor branch, method entries
    and panel title.
    """
    panel = accessor_with_rich._create_help_panel()
    out = _render_panel(panel)
    # Check for title, root node (parent class), accessor name, and method
    for expected in (
        "Mock accessor",
        accessor_with_rich._parent.__class__.__name__,
        "mock_accessor",
        "fetch",
    ):
        assert expected in out


def test_rich_display_help_mixin_tree_and_title(display_with_rich):
    """_RichHelpDisplayMixin tree contains Methods, Attributes and panel title."""
    panel = display_with_rich._create_help_panel()
    out = _render_panel(panel)
    for expected in (
        "Mock display",
        "Methods",
        "Attributes",
        "some_attr",
        display_with_rich.__class__.__name__,
    ):
        assert expected in out
