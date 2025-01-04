import inspect
from io import StringIO

from rich.console import Console
from rich.tree import Tree


class HelpDisplayMixin:
    """Mixin class to add help functionality to a class."""

    def _get_attributes_for_help(self):
        """Get the attributes ending with '_' to display in help."""
        attributes = []
        for name in dir(self):
            if name.endswith("_") and not name.startswith("_"):
                attributes.append(name)
        return sorted(attributes)

    def _get_methods_for_help(self):
        """Get the public methods to display in help."""
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        filtered_methods = []
        for name, method in methods:
            is_private = name.startswith("_")
            is_class_method = inspect.ismethod(method) and method.__self__ is type(self)
            is_help_method = name == "help"
            if not (is_private or is_class_method or is_help_method):
                filtered_methods.append((name, method))
        return sorted(filtered_methods)

    def _create_help_tree(self):
        """Create a rich Tree with attributes and methods."""
        tree = Tree(f"📊 {self.__class__.__name__}")

        attributes = self._get_attributes_for_help()
        if attributes:
            attr_branch = tree.add("📝 Attributes to tweak your plot")
            # Ensure figure_ and ax_ are first
            sorted_attrs = sorted(attributes)
            if "ax_" in sorted_attrs:
                sorted_attrs.remove("ax_")
            if "figure_" in sorted_attrs:
                sorted_attrs.remove("figure_")
            sorted_attrs = ["figure_", "ax_"] + [
                attr for attr in sorted_attrs if attr not in ["figure_", "ax_"]
            ]
            for attr in sorted_attrs:
                attr_branch.add(attr)

        methods = self._get_methods_for_help()
        if methods:
            method_branch = tree.add("🔧 Methods")
            for name, method in methods:
                description = (
                    method.__doc__.split("\n")[0]
                    if method.__doc__
                    else "No description available"
                )
                method_branch.add(f"{name} - {description}")

        return tree

    def help(self):
        """Display available attributes and methods using rich."""
        from skore import console  # avoid circular import

        console.print(self._create_help_tree())

    def __repr__(self):
        """Return a string representation using rich."""
        console = Console(file=StringIO(), force_terminal=False)
        console.print(self._create_help_tree())
        return console.file.getvalue()


def _despine_matplotlib_axis(ax):
    """Despine the matplotlib axis."""
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]:
        ax.spines[s].set_bounds(0, 1)
