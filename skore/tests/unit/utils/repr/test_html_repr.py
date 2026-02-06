"""Unit tests for ``skore._utils.repr.html_repr``."""

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
