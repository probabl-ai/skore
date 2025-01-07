"""Custom help formatter for the CLI."""

import re
import shutil
from argparse import ArgumentParser, HelpFormatter

from rich.console import Console
from rich.theme import Theme

skore_console_theme = Theme(
    {
        "repr.str": "cyan",
        "rule.line": "orange1",
        "repr.url": "orange1",
    }
)


class RichColorHelpFormatter(HelpFormatter):
    """Custom help formatter for the CLI."""

    def __init__(self, prog, indent_increment=2, max_help_position=24, width=None):
        width = shutil.get_terminal_size()[0] if width is None else width
        super().__init__(prog, indent_increment, max_help_position, width)
        self.console = Console(theme=skore_console_theme)

    def _format_action_invocation(self, action):
        """Format the action invocation (flags and arguments)."""
        if not action.option_strings:
            metavar = self._metavar_formatter(action, action.dest)(1)[0]
            return metavar
        else:
            parts = []
            # Format short options
            if action.option_strings:
                parts.extend(
                    f"[cyan bold]{opt}[/cyan bold]" for opt in action.option_strings
                )
            # Format argument
            if action.nargs != 0:
                default = self._get_default_metavar_for_optional(action)
                args_string = self._format_args(action, default)
                parts.append(f"[orange1 bold]{args_string}[/orange1 bold]")

            return " ".join(parts)

    def _format_usage(self, usage, actions, groups, prefix):
        """Format the usage line."""
        if prefix is None:
            prefix = "usage: "

        # Format the usage line
        formatted = super()._format_usage(usage, actions, groups, prefix)

        # Apply rich formatting
        formatted = re.sub(r"usage:", "[orange1 bold]usage:[/orange1 bold]", formatted)
        formatted = re.sub(
            r"(?<=\[)[A-Z_]+(?=\])", lambda m: f"[cyan]{m.group()}[/cyan]", formatted
        )

        return formatted

    def format_help(self):
        """Format the help message."""
        help_text = super().format_help()

        # Format section headers
        help_text = re.sub(
            r"^([a-zA-Z ]+ arguments:)$",
            r"[dim]\1[/dim]",
            help_text,
            flags=re.MULTILINE,
        )

        # Format default values
        help_text = re.sub(
            r"\(default: .*?\)", lambda m: f"[dim]{m.group()}[/dim]", help_text
        )

        # Color the subcommands in cyan
        help_text = re.sub(
            r"(?<=\s)(launch|create|quickstart)(?=\s+)",
            r"[cyan bold]\1[/cyan bold]",
            help_text,
        )

        # Color "options" in orange1
        help_text = re.sub(
            r"(?<=\s)(options)(?=:)",
            r"[orange1]\1[/orange1]",
            help_text,
        )

        return help_text


class ColorArgumentParser(ArgumentParser):
    """Custom argument parser for the CLI."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, formatter_class=RichColorHelpFormatter)

    def print_help(self, file=None):
        """Print the help message."""
        console = Console(file=file)
        console.print(self.format_help())
