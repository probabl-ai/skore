"""Unit tests for ``skore._utils.repr.html_repr``."""

from typing import ClassVar

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from skore._utils._testing import MockAccessor, MockDisplay, MockReport
from skore._utils.repr.html_repr import (
    _HTMLAccessorHelpMixin,
    _HTMLHelpDisplayMixin,
    _HTMLReportHelpMixin,
)


class _ReportWithHTML(MockReport, _HTMLReportHelpMixin):
    """Minimal report with HTML help mixin for tests."""


class _AccessorWithHTML(MockAccessor, _HTMLAccessorHelpMixin):
    """Minimal accessor with HTML help mixin for tests."""

    def _get_help_title(self) -> str:
        return "Mock accessor"


class _GroupedAccessorWithHTML(MockAccessor, _HTMLAccessorHelpMixin):
    """Accessor declaring `_HELP_METHOD_GROUPS` with HTML help mixin."""

    _HELP_METHOD_GROUPS: ClassVar[dict[str, tuple[str, ...]]] = {
        "Registry": ("alpha", "beta"),
        "Metrics": ("gamma",),
        "Displays": ("epsilon",),
    }

    def alpha(self):
        """Alpha method."""

    def beta(self):
        """Beta method."""

    def gamma(self):
        """Gamma method."""

    def epsilon(self):
        """Epsilon method."""

    def _get_help_title(self) -> str:
        return "Grouped accessor"


class _ReportWithGroupedHTML(MockReport, _HTMLReportHelpMixin):
    """Report whose accessor declares `_HELP_METHOD_GROUPS`."""

    _ACCESSOR_CONFIG = {"metrics": {"name": "metrics"}}

    def __init__(self, estimator, X_train=None, y_train=None, X_test=None, y_test=None):
        super().__init__(
            estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )
        self.metrics = _GroupedAccessorWithHTML(parent=self)


class _DisplayWithHTML(MockDisplay, _HTMLHelpDisplayMixin):
    """Minimal display with HTML help mixin for tests."""

    def _get_help_title(self) -> str:
        return "Mock display"


@pytest.fixture
def report_with_html():
    """Report with HTML help mixin."""
    X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y = np.array([0, 1, 1, 0])
    estimator = LogisticRegression().fit(X, y)
    return _ReportWithHTML(estimator)


def test_html_report_help_mixin_creates_container(report_with_html):
    """_HTMLReportHelpMixin._create_help_html returns HTML containing the container
    id.
    """
    html = report_with_html._create_help_html()
    assert "skore-help-" in html


def test_html_accessor_help_mixin_creates_container(report_with_html):
    """_HTMLAccessorHelpMixin._create_help_html returns HTML containing the container
    id.
    """
    accessor = _AccessorWithHTML(parent=report_with_html)
    html = accessor._create_help_html()
    assert "skore-accessor-help-" in html


def test_html_display_help_mixin_creates_container():
    """_HTMLHelpDisplayMixin._create_help_html returns HTML containing the container
    id.
    """
    display = _DisplayWithHTML()
    html = display._create_help_html()
    assert "skore-display-help-" in html


def test_html_accessor_help_mixin_renders_groups(report_with_html):
    """Grouped accessor HTML help renders Registry / Metrics / Displays sections."""
    accessor = _GroupedAccessorWithHTML(parent=report_with_html)
    html = accessor._create_help_html()
    for expected in ("Registry", "Metrics", "Displays", "alpha", "gamma", "epsilon"):
        assert expected in html
    # Method groups are unfolded by default.
    assert html.count('<input type="checkbox" class="toggle" checked>') >= 3


def test_html_report_help_mixin_renders_groups():
    """Report HTML help propagates accessor method groups."""
    X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y = np.array([0, 1, 1, 0])
    estimator = LogisticRegression().fit(X, y)
    report = _ReportWithGroupedHTML(estimator)
    html = report._create_help_html()
    for expected in ("Registry", "Metrics", "Displays", "alpha", "gamma", "epsilon"):
        assert expected in html
