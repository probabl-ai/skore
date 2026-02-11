"""HTML-based help rendering mixins."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from skore._utils.repr.data import (
    _AccessorHelpDataMixin,
    _DisplayHelpDataMixin,
    _ReportHelpDataMixin,
)


def get_jinja_env():
    """Get Jinja2 environment for loading templates."""
    template_dir = Path(__file__).parent
    return Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)


class _BaseHTMLHelpMixin(ABC):
    """Base mixin for HTML-based help rendering."""

    @abstractmethod
    def _create_help_html(self) -> str:
        """Create the HTML representation of the help tree."""


class _HTMLReportHelpMixin(_ReportHelpDataMixin, _BaseHTMLHelpMixin):
    """Mixin for HTML-based help rendering for reports with Shadow DOM isolation."""

    def _create_help_html(self) -> str:
        """Create the HTML representation of the help tree."""
        template_data = asdict(self._build_help_data())
        template_data["is_report"] = True

        env = get_jinja_env()
        template = env.get_template("report_help.html.j2")

        container_id = f"skore-help-{uuid.uuid4().hex[:8]}"

        shadow_dom_html = template.render(
            container_id=container_id,
            **template_data,
        )

        return shadow_dom_html


class _HTMLAccessorHelpMixin(_AccessorHelpDataMixin, _BaseHTMLHelpMixin):
    """Mixin for HTML-based help rendering for accessors with Shadow DOM isolation."""

    def _create_help_html(self) -> str:
        """Create the HTML representation of the help tree for accessors."""
        template_data = asdict(self._build_help_data())
        template_data["is_report"] = False

        env = get_jinja_env()
        template = env.get_template("report_help.html.j2")

        container_id = f"skore-accessor-help-{uuid.uuid4().hex[:8]}"

        shadow_dom_html = template.render(
            container_id=container_id,
            **template_data,
        )

        return shadow_dom_html


class _HTMLHelpDisplayMixin(_DisplayHelpDataMixin, _BaseHTMLHelpMixin):
    """Mixin for HTML-based help rendering for displays with Shadow DOM isolation."""

    def _create_help_html(self) -> str:
        """Create the HTML representation of the help tree."""
        template_data = asdict(self._build_help_data())

        env = get_jinja_env()
        template = env.get_template("display_help.html.j2")

        container_id = f"skore-display-help-{uuid.uuid4().hex[:8]}"

        shadow_dom_html = template.render(
            container_id=container_id,
            **template_data,
        )

        return shadow_dom_html
