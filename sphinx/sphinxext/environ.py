"""Custom Sphinx extension to insert environment variables into RST files."""
from os import environ

from docutils import nodes
from docutils.parsers.rst import Directive


class EnvironDirective(Directive):
    """
    Directive to insert the value of an environment variable into the document.

    Example usage in RST

        Refer to the `example <.. environ:: VAR>`_.
    """
    required_arguments = 1

    def run(self):
        var_name = self.arguments[0]
        value = environ.get(var_name, "<missing>")
        return [nodes.Text(value)]


def setup(app):
    """Setup the extension."""
    app.add_directive('environ', EnvironDirective)
    return {'version': '1.0'}
