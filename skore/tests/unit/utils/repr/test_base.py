"""Unit tests for ``skore._utils.repr.base``."""

from unittest.mock import Mock

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


def test_report_help_mixin_sphinx_path(monkeypatch, report_with_base_help):
    """ReportHelpMixin.help returns _HelpDisplay when sphinx build."""
    mock_sphinx = Mock(return_value=True)
    mock_notebook = Mock(return_value=False)
    monkeypatch.setattr(
        "skore._utils.repr.base.is_environment_sphinx_build", mock_sphinx
    )
    monkeypatch.setattr(
        "skore._utils.repr.base.is_environment_notebook_like", mock_notebook
    )
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


def test_report_help_mixin_html_path(monkeypatch, report_with_base_help):
    """ReportHelpMixin.help uses HTML path when notebook-like."""
    sentinel_html = object()
    mock_html_cls = Mock(return_value=sentinel_html)
    mock_display = Mock()
    mock_notebook = Mock(return_value=True)
    monkeypatch.setattr(
        "skore._utils.repr.base.is_environment_notebook_like", mock_notebook
    )
    monkeypatch.setattr("IPython.display.display", mock_display)
    monkeypatch.setattr("IPython.display.HTML", mock_html_cls)
    report_with_base_help.help()
    mock_notebook.assert_called_once()
    mock_html_cls.assert_called_once()
    html_arg = mock_html_cls.call_args[0][0]
    assert "skore-help-" in html_arg
    mock_display.assert_called_once_with(sentinel_html)


def test_report_help_mixin_rich_path(monkeypatch, report_with_base_help):
    """ReportHelpMixin.help uses Rich path when not notebook-like."""
    mock_console_print = Mock()
    mock_notebook = Mock(return_value=False)
    monkeypatch.setattr(
        "skore._utils.repr.base.is_environment_notebook_like", mock_notebook
    )
    monkeypatch.setattr("skore.console.print", mock_console_print)
    report_with_base_help.help()
    mock_notebook.assert_called_once()
    mock_console_print.assert_called_once()
    (panel,) = mock_console_print.call_args[0]
    assert panel is not None


def test_accessor_help_mixin_sphinx_path(monkeypatch, accessor_with_base_help):
    """AccessorHelpMixin.help returns _HelpDisplay when sphinx build."""
    mock_sphinx = Mock(return_value=True)
    mock_notebook = Mock(return_value=False)
    monkeypatch.setattr(
        "skore._utils.repr.base.is_environment_sphinx_build", mock_sphinx
    )
    monkeypatch.setattr(
        "skore._utils.repr.base.is_environment_notebook_like", mock_notebook
    )
    result = accessor_with_base_help.help()
    mock_sphinx.assert_called_once()
    mock_notebook.assert_not_called()
    assert result is not None
    assert isinstance(result, _HelpDisplay)
    assert "skore-accessor-help-" in result._repr_html_()
    bundle = result._repr_mimebundle_()
    assert "text/html" in bundle
    assert "text/plain" in bundle


def test_accessor_help_mixin_html_path(monkeypatch, accessor_with_base_help):
    """AccessorHelpMixin.help uses HTML path when notebook-like."""
    sentinel_html = object()
    mock_html_cls = Mock(return_value=sentinel_html)
    mock_display = Mock()
    mock_notebook = Mock(return_value=True)
    monkeypatch.setattr(
        "skore._utils.repr.base.is_environment_notebook_like", mock_notebook
    )
    monkeypatch.setattr("IPython.display.display", mock_display)
    monkeypatch.setattr("IPython.display.HTML", mock_html_cls)
    accessor_with_base_help.help()
    mock_notebook.assert_called_once()
    mock_html_cls.assert_called_once()
    html_arg = mock_html_cls.call_args[0][0]
    assert "skore-accessor-help-" in html_arg
    mock_display.assert_called_once_with(sentinel_html)


def test_accessor_help_mixin_rich_path(monkeypatch, accessor_with_base_help):
    """AccessorHelpMixin.help uses Rich path when not notebook-like."""
    mock_console_print = Mock()
    mock_notebook = Mock(return_value=False)
    monkeypatch.setattr(
        "skore._utils.repr.base.is_environment_notebook_like", mock_notebook
    )
    monkeypatch.setattr("skore.console.print", mock_console_print)
    accessor_with_base_help.help()
    mock_notebook.assert_called_once()
    mock_console_print.assert_called_once()
    (panel,) = mock_console_print.call_args[0]
    assert panel is not None


def test_display_help_mixin_sphinx_path(monkeypatch, display_with_base_help):
    """DisplayHelpMixin.help returns _HelpDisplay when sphinx build."""
    mock_sphinx = Mock(return_value=True)
    mock_notebook = Mock(return_value=False)
    monkeypatch.setattr(
        "skore._utils.repr.base.is_environment_sphinx_build", mock_sphinx
    )
    monkeypatch.setattr(
        "skore._utils.repr.base.is_environment_notebook_like", mock_notebook
    )
    result = display_with_base_help.help()
    mock_sphinx.assert_called_once()
    mock_notebook.assert_not_called()
    assert result is not None
    assert isinstance(result, _HelpDisplay)
    assert "skore-display-help-" in result._repr_html_()
    bundle = result._repr_mimebundle_()
    assert "text/html" in bundle
    assert "text/plain" in bundle


def test_display_help_mixin_html_path(monkeypatch, display_with_base_help):
    """DisplayHelpMixin.help uses HTML path when notebook-like."""
    sentinel_html = object()
    mock_html_cls = Mock(return_value=sentinel_html)
    mock_display = Mock()
    mock_notebook = Mock(return_value=True)
    monkeypatch.setattr(
        "skore._utils.repr.base.is_environment_notebook_like", mock_notebook
    )
    monkeypatch.setattr("IPython.display.display", mock_display)
    monkeypatch.setattr("IPython.display.HTML", mock_html_cls)
    display_with_base_help.help()
    mock_notebook.assert_called_once()
    mock_html_cls.assert_called_once()
    html_arg = mock_html_cls.call_args[0][0]
    assert "skore-display-help-" in html_arg
    mock_display.assert_called_once_with(sentinel_html)


def test_display_help_mixin_rich_path(monkeypatch, display_with_base_help):
    """DisplayHelpMixin.help uses Rich path when not notebook-like."""
    mock_console_print = Mock()
    mock_notebook = Mock(return_value=False)
    monkeypatch.setattr(
        "skore._utils.repr.base.is_environment_notebook_like", mock_notebook
    )
    monkeypatch.setattr("skore.console.print", mock_console_print)
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
