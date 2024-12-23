import inspect
from io import StringIO

from rich.console import Console
from rich.tree import Tree


class _HelpMixin:
    """Mixin class providing help for the `help` method and the `__repr__` method."""

    def _get_methods_for_help(self):
        """Get the methods to display in help."""
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        filtered_methods = []
        for name, method in methods:
            is_private_method = name.startswith("_")
            # we cannot use `isinstance(method, classmethod)` because it is already
            # already transformed by the decorator `@classmethod`.
            is_class_method = inspect.ismethod(method) and method.__self__ is type(self)
            is_help_method = name == "help"
            if not (is_private_method or is_class_method or is_help_method):
                filtered_methods.append((name, method))
        return filtered_methods

    def _sort_methods_for_help(self, methods):
        """Sort methods for help display."""
        return sorted(methods)

    def _format_method_name(self, name):
        """Format method name for display."""
        return name

    def _get_method_description(self, method):
        """Get the description for a method."""
        return (
            method.__doc__.split("\n")[0]
            if method.__doc__
            else "No description available"
        )

    def help(self):
        """Display available methods using rich."""
        from skore import console  # avoid circular import

        console.print(self._create_help_tree())

    def __repr__(self):
        """Return a string representation using rich."""
        console = Console(file=StringIO(), force_terminal=False)
        console.print(self._create_help_tree())
        return console.file.getvalue()


class _HelpReportMixin(_HelpMixin):
    _ACCESSOR_CONFIG = {
        "plot": {"icon": "🎨", "name": "plot"},
        "metrics": {"icon": "📏", "name": "metrics"},
        # Add other accessors as they're implemented
        # "inspection": {"icon": "🔍", "name": "inspection"},
        # "linting": {"icon": "✔️", "name": "linting"},
    }

    def _get_help_title(self):
        return f"🔧 Available tools for diagnosing {self.estimator_name} estimator"

    def _create_help_tree(self):
        """Create a rich Tree with the available tools and accessor methods."""
        tree = Tree(self._get_help_title())

        # Add accessor methods first
        for accessor_attr, config in self._ACCESSOR_CONFIG.items():
            if hasattr(self, accessor_attr):
                accessor = getattr(self, accessor_attr)
                branch = tree.add(f"{config['icon']} {config['name']}")

                # Use accessor's _get_methods_for_help instead of inspect.getmembers
                methods = accessor._get_methods_for_help()
                methods = accessor._sort_methods_for_help(methods)

                # Add methods to branch
                for name, method in methods:
                    displayed_name = accessor._format_method_name(name)
                    description = accessor._get_method_description(method)
                    branch.add(f"{displayed_name} - {description}")

        # Add base methods last
        base_methods = self._get_methods_for_help()
        base_methods = self._sort_methods_for_help(base_methods)

        for name, method in base_methods:
            if not hasattr(getattr(self, name), "_icon"):  # Skip accessors
                description = self._get_method_description(method)
                tree.add(f"{name} - {description}")

        return tree


class _HelpAccessorMixin(_HelpMixin):
    def _get_help_title(self):
        name = self.__class__.__name__.replace("_", "").replace("Accessor", "").lower()
        return f"{self._icon} Available {name} methods"

    def _create_help_tree(self):
        """Create a rich Tree with the available methods."""
        tree = Tree(self._get_help_title())

        methods = self._get_methods_for_help()
        methods = self._sort_methods_for_help(methods)

        for name, method in methods:
            if not name.startswith("_"):
                displayed_name = self._format_method_name(name)
                description = self._get_method_description(method)
                tree.add(f"{displayed_name} - {description}")

        return tree
