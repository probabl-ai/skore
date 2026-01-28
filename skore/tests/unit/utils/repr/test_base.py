"""Unit tests for ``skore._utils.repr.base``."""

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from skore._utils._testing import MockAccessor, MockDisplay, MockReport
from skore._utils.repr.base import (
    AccessorHelpMixin,
    DisplayHelpMixin,
    ReportHelpMixin,
    ReprHTMLMixin,
    _HelpDisplay,
)


class _ReportWithBaseHelp(MockReport, ReportHelpMixin):
    """Minimal report with ReportHelpMixin for tests."""


class _AccessorWithBaseHelp(MockAccessor, AccessorHelpMixin):
    """Minimal accessor with AccessorHelpMixin for tests."""


class _DisplayWithBaseHelp(DisplayHelpMixin, MockDisplay):
    """Minimal display with DisplayHelpMixin for tests.

    DisplayHelpMixin must come before MockDisplay so help() is the mixin's
    implementation; MockDisplay.help just passes and would override otherwise.
    """

    estimator_name = "Mock"


class _WithReprHTML(ReprHTMLMixin):
    """Minimal class with ReprHTMLMixin for tests."""

    def _html_repr(self):
        return "<p>test</p>"


@pytest.fixture
def report_with_base_help():
    """Report with ReportHelpMixin."""
    X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y = np.array([0, 1, 1, 0])
    estimator = LogisticRegression().fit(X, y)
    return _ReportWithBaseHelp(estimator)


@pytest.fixture
def accessor_with_base_help(report_with_base_help):
    """Accessor with AccessorHelpMixin."""
    return _AccessorWithBaseHelp(parent=report_with_base_help)


@pytest.fixture
def display_with_base_help():
    """Display with DisplayHelpMixin."""
    return _DisplayWithBaseHelp()


@patch("skore._utils.repr.base.is_environment_sphinx_build", return_value=True)
@patch("skore._utils.repr.base.is_environment_notebook_like", return_value=False)
def test_report_help_mixin_sphinx_path(
    mock_notebook, mock_sphinx, report_with_base_help
):
    """ReportHelpMixin.help returns _HelpDisplay when sphinx build."""
    result = report_with_base_help.help()
    mock_sphinx.assert_called_once()
    mock_notebook.assert_not_called()
    assert result is not None
    assert isinstance(result, _HelpDisplay)
    assert "skore-help-" in result._repr_html_()
    bundle = result._repr_mimebundle_()
    assert "text/html" in bundle
    assert "text/plain" in bundle
    assert "skore-help-" in bundle["text/html"]


@patch("skore._utils.repr.base.is_environment_notebook_like", return_value=True)
@patch("IPython.display.display")
@patch("IPython.display.HTML")
def test_report_help_mixin_html_path(
    mock_html_cls, mock_display, mock_notebook, report_with_base_help
):
    """ReportHelpMixin.help uses HTML path when notebook-like."""
    mock_html_cls.return_value = sentinel_html = object()
    report_with_base_help.help()
    mock_notebook.assert_called_once()
    mock_html_cls.assert_called_once()
    html_arg = mock_html_cls.call_args[0][0]
    assert "skore-help-" in html_arg
    mock_display.assert_called_once_with(sentinel_html)


@patch("skore._utils.repr.base.is_environment_notebook_like", return_value=False)
@patch("skore.console.print")
def test_report_help_mixin_rich_path(
    mock_console_print, mock_notebook, report_with_base_help
):
    """ReportHelpMixin.help uses Rich path when not notebook-like."""
    report_with_base_help.help()
    mock_notebook.assert_called_once()
    mock_console_print.assert_called_once()
    (panel,) = mock_console_print.call_args[0]
    assert panel is not None


@patch("skore._utils.repr.base.is_environment_sphinx_build", return_value=True)
@patch("skore._utils.repr.base.is_environment_notebook_like", return_value=False)
def test_accessor_help_mixin_sphinx_path(
    mock_notebook, mock_sphinx, accessor_with_base_help
):
    """AccessorHelpMixin.help returns _HelpDisplay when sphinx build."""
    result = accessor_with_base_help.help()
    mock_sphinx.assert_called_once()
    mock_notebook.assert_not_called()
    assert result is not None
    assert isinstance(result, _HelpDisplay)
    assert "skore-accessor-help-" in result._repr_html_()
    bundle = result._repr_mimebundle_()
    assert "text/html" in bundle
    assert "text/plain" in bundle


@patch("skore._utils.repr.base.is_environment_notebook_like", return_value=True)
@patch("IPython.display.display")
@patch("IPython.display.HTML")
def test_accessor_help_mixin_html_path(
    mock_html_cls, mock_display, mock_notebook, accessor_with_base_help
):
    """AccessorHelpMixin.help uses HTML path when notebook-like."""
    mock_html_cls.return_value = sentinel_html = object()
    accessor_with_base_help.help()
    mock_notebook.assert_called_once()
    mock_html_cls.assert_called_once()
    html_arg = mock_html_cls.call_args[0][0]
    assert "skore-accessor-help-" in html_arg
    mock_display.assert_called_once_with(sentinel_html)


@patch("skore._utils.repr.base.is_environment_notebook_like", return_value=False)
@patch("skore.console.print")
def test_accessor_help_mixin_rich_path(
    mock_console_print, mock_notebook, accessor_with_base_help
):
    """AccessorHelpMixin.help uses Rich path when not notebook-like."""
    accessor_with_base_help.help()
    mock_notebook.assert_called_once()
    mock_console_print.assert_called_once()
    (panel,) = mock_console_print.call_args[0]
    assert panel is not None


@patch("skore._utils.repr.base.is_environment_sphinx_build", return_value=True)
@patch("skore._utils.repr.base.is_environment_notebook_like", return_value=False)
def test_display_help_mixin_sphinx_path(
    mock_notebook, mock_sphinx, display_with_base_help
):
    """DisplayHelpMixin.help returns _HelpDisplay when sphinx build."""
    result = display_with_base_help.help()
    mock_sphinx.assert_called_once()
    mock_notebook.assert_not_called()
    assert result is not None
    assert isinstance(result, _HelpDisplay)
    assert "skore-display-help-" in result._repr_html_()
    bundle = result._repr_mimebundle_()
    assert "text/html" in bundle
    assert "text/plain" in bundle


@patch("skore._utils.repr.base.is_environment_notebook_like", return_value=True)
@patch("IPython.display.display")
@patch("IPython.display.HTML")
def test_display_help_mixin_html_path(
    mock_html_cls, mock_display, mock_notebook, display_with_base_help
):
    """DisplayHelpMixin.help uses HTML path when notebook-like."""
    mock_html_cls.return_value = sentinel_html = object()
    display_with_base_help.help()
    mock_notebook.assert_called_once()
    mock_html_cls.assert_called_once()
    html_arg = mock_html_cls.call_args[0][0]
    assert "skore-display-help-" in html_arg
    mock_display.assert_called_once_with(sentinel_html)


@patch("skore._utils.repr.base.is_environment_notebook_like", return_value=False)
@patch("skore.console.print")
def test_display_help_mixin_rich_path(
    mock_console_print, mock_notebook, display_with_base_help
):
    """DisplayHelpMixin.help uses Rich path when not notebook-like."""
    display_with_base_help.help()
    mock_notebook.assert_called_once()
    mock_console_print.assert_called_once()
    (panel,) = mock_console_print.call_args[0]
    assert panel is not None


def test_repr_html_mixin_repr_html_inner():
    """ReprHTMLMixin._repr_html_ delegates to _html_repr via _repr_html_inner."""
    obj = _WithReprHTML()
    assert obj._repr_html_ == obj._repr_html_inner
    assert obj._repr_html_inner() == "<p>test</p>"


def test_repr_html_mixin_repr_mimebundle():
    """ReprHTMLMixin._repr_mimebundle_ returns text/plain and text/html."""
    obj = _WithReprHTML()
    out = obj._repr_mimebundle_()
    assert "text/plain" in out
    assert "text/html" in out
    assert out["text/html"] == "<p>test</p>"
    assert repr(obj) in out["text/plain"]
