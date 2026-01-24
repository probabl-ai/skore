from skore._utils._environment import is_environment_notebook_like
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


class ReportHelpMixin(_RichReportHelpMixin, _HTMLReportHelpMixin):
    """Mixin class providing help for report `help` and `__repr__`.

    This mixin inherits from both `_RichHelpMixin` and `_HTMLHelpMixin` and
    delegates to the appropriate implementation based on the environment.
    """

    def help(self) -> None:
        """Display report help using rich or HTML."""
        if is_environment_notebook_like():
            from IPython.display import HTML, display

            display(HTML(self._create_help_html()))
        else:
            from skore import console  # avoid circular import

            console.print(self._create_help_panel())


class AccessorHelpMixin(_RichAccessorHelpMixin, _HTMLAccessorHelpMixin):
    """Mixin class providing help for accessor `help`."""

    def _get_help_title(self) -> str:
        name = self.__class__.__name__.replace("_", "").replace("Accessor", "").lower()
        return f"{name.capitalize()} accessor"

    def help(self) -> None:
        """Display accessor help using rich or HTML."""
        if is_environment_notebook_like():
            from IPython.display import HTML, display

            display(HTML(self._create_help_html()))
        else:
            from skore import console  # avoid circular import

            console.print(self._create_help_panel())


class DisplayHelpMixin(_RichHelpDisplayMixin, _HTMLHelpDisplayMixin):
    """Mixin class providing help for display `help` and `__repr__`.

    This mixin inherits from both `_RichHelpDisplayMixin` and `_HTMLHelpDisplayMixin`
    and delegates to the appropriate implementation based on the environment.
    """

    estimator_name: str  # defined in the concrete display class

    def _get_help_title(self) -> str:
        """Get the help title for the display."""
        return f"{self.__class__.__name__} display"

    def help(self) -> None:
        """Display display help using rich or HTML."""
        if is_environment_notebook_like():
            from IPython.display import HTML, display

            display(HTML(self._create_help_html()))
        else:
            from skore import console  # avoid circular import

            console.print(self._create_help_panel())


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
