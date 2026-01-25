from __future__ import annotations

from io import StringIO
from typing import Callable

from rich.console import Console
from rich.panel import Panel

from skore._utils._environment import (
    is_environment_notebook_like,
    is_environment_sphinx_build,
)
from skore._utils.repr.html_repr import (
    _HTMLAccessorHelpMixin,
    _HTMLHelpDisplayMixin,
    _HTMLReportHelpMixin,
)
from skore._utils.repr.rich_repr import (
    _RichAccessorHelpMixin,
    _RichHelpDisplayMixin,
    _RichReportHelpMixin,
)


class _HelpDisplay:
    """Displayable help with ``_repr_html_`` and ``_repr_mimebundle_`` for sphinx-gallery."""

    __slots__ = ("_html", "_plain")

    def __init__(self, *, html: str, plain: str) -> None:
        self._html = html
        self._plain = plain

    def _repr_html_(self) -> str:
        return self._html

    def _repr_mimebundle_(self, **kwargs: object) -> dict[str, str]:
        return {"text/plain": self._plain, "text/html": self._html}


def _render_panel_to_plain_text(panel: Panel) -> str:
    """Render a Rich Panel to a plain string (e.g. for mimebundle text/plain)."""
    buf = StringIO()
    Console(file=buf, force_terminal=False).print(panel)
    return buf.getvalue()


def _help_dispatch(
    *,
    html_help_factory: Callable[[], str],
    rich_help_factory: Callable[[], Panel],
) -> _HelpDisplay | None:
    """Dispatch help to sphinx (mimebundle), notebook (display HTML), or terminal (Rich).

    Each specialized ``help()`` calls this with ``_create_help_html`` and
    ``_create_help_panel``. Behaviour is unchanged.
    """
    if is_environment_sphinx_build():
        return _HelpDisplay(
            html=html_help_factory(),
            plain=_render_panel_to_plain_text(rich_help_factory()),
        )
    if is_environment_notebook_like():
        from IPython.display import HTML, display

        display(HTML(html_help_factory()))
        return None
    from skore import console  # avoid circular import

    console.print(rich_help_factory())
    return None


class ReportHelpMixin(_RichReportHelpMixin, _HTMLReportHelpMixin):
    """Mixin class providing help for report `help` and `__repr__`.

    This mixin inherits from both `_RichHelpMixin` and `_HTMLHelpMixin` and
    delegates to the appropriate implementation based on the environment.
    """

    def help(self) -> _HelpDisplay | None:
        """Display report help using rich or HTML."""
        return _help_dispatch(
            html_help_factory=self._create_help_html,
            rich_help_factory=self._create_help_panel,
        )


class AccessorHelpMixin(_RichAccessorHelpMixin, _HTMLAccessorHelpMixin):
    """Mixin class providing help for accessor `help`."""

    def _get_help_title(self) -> str:
        name = self.__class__.__name__.replace("_", "").replace("Accessor", "").lower()
        return f"{name.capitalize()} accessor"

    def help(self) -> _HelpDisplay | None:
        """Display accessor help using rich or HTML."""
        return _help_dispatch(
            html_help_factory=self._create_help_html,
            rich_help_factory=self._create_help_panel,
        )


class DisplayHelpMixin(_RichHelpDisplayMixin, _HTMLHelpDisplayMixin):
    """Mixin class providing help for display `help` and `__repr__`.

    This mixin inherits from both `_RichHelpDisplayMixin` and `_HTMLHelpDisplayMixin`
    and delegates to the appropriate implementation based on the environment.
    """

    estimator_name: str  # defined in the concrete display class

    def _get_help_title(self) -> str:
        """Get the help title for the display."""
        return f"{self.__class__.__name__} display"

    def help(self) -> _HelpDisplay | None:
        """Display display help using rich or HTML."""
        return _help_dispatch(
            html_help_factory=self._create_help_html,
            rich_help_factory=self._create_help_panel,
        )


class ReprHTMLMixin:
    """Mixin to handle consistently the HTML representation.

    When inheriting from this class, you need to define an attribute `_html_repr`
    which is a callable that returns the HTML representation to be shown.
    """

    @property
    def _repr_html_(self):
        return self._repr_html_inner

    def _repr_html_inner(self):
        return self._html_repr()

    def _repr_mimebundle_(self, **kwargs):
        return {"text/plain": repr(self), "text/html": self._html_repr()}
