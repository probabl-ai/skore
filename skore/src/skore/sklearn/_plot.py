import inspect
from io import StringIO

from rich.console import Console
from rich.tree import Tree
from sklearn.metrics import (
    PrecisionRecallDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
)


def _despine_matplotlib_axis(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]:
        ax.spines[s].set_bounds(0, 1)


class HelpDisplayMixin:
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


class RocCurveDisplay(HelpDisplayMixin, RocCurveDisplay):
    def plot(
        self,
        ax=None,
        *,
        name=None,
        plot_chance_level=False,
        chance_level_kw=None,
        despine=False,
    ):
        """Plot visualization.

        Extra keyword arguments will be passed to matplotlib's ``plot``.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        name : str, default=None
            Name of ROC Curve for labeling. If `None`, use `estimator_name` if
            not `None`, otherwise no labeling is shown.

        plot_chance_level : bool, default=False
            Whether to plot the chance level.

        chance_level_kw : dict, default=None
            Keyword arguments to be passed to matplotlib's `plot` for rendering
            the chance level line.

        despine : bool, default=False
            Whether to remove the top and right spines from the plot.

        **kwargs : dict
            Keyword arguments to be passed to matplotlib's `plot`.

        Returns
        -------
        display : :class:`~sklearn.metrics.RocCurveDisplay`
            Object that stores computed values.
        """
        super().plot(
            ax=ax,
            name=name,
            plot_chance_level=plot_chance_level,
            chance_level_kw=chance_level_kw,
        )
        if despine:
            _despine_matplotlib_axis(self.ax_)

        return self


class PrecisionRecallDisplay(HelpDisplayMixin, PrecisionRecallDisplay):
    def plot(
        self,
        ax=None,
        *,
        name=None,
        plot_chance_level=False,
        chance_level_kw=None,
        despine=False,
    ):
        """Plot visualization.

        Extra keyword arguments will be passed to matplotlib's `plot`.

        Parameters
        ----------
        ax : Matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        name : str, default=None
            Name of precision recall curve for labeling. If `None`, use
            `estimator_name` if not `None`, otherwise no labeling is shown.

        plot_chance_level : bool, default=False
            Whether to plot the chance level. The chance level is the prevalence
            of the positive label computed from the data passed during
            :meth:`from_estimator` or :meth:`from_predictions` call.

        chance_level_kw : dict, default=None
            Keyword arguments to be passed to matplotlib's `plot` for rendering
            the chance level line.

        despine : bool, default=False
            Whether to remove the top and right spines from the plot.

        **kwargs : dict
            Keyword arguments to be passed to matplotlib's `plot`.

        Returns
        -------
        display : :class:`~sklearn.metrics.PrecisionRecallDisplay`
            Object that stores computed values.

        Notes
        -----
        The average precision (cf. :func:`~sklearn.metrics.average_precision_score`)
        in scikit-learn is computed without any interpolation. To be consistent
        with this metric, the precision-recall curve is plotted without any
        interpolation as well (step-wise style).

        You can change this style by passing the keyword argument
        `drawstyle="default"`. However, the curve will not be strictly
        consistent with the reported average precision.
        """
        super().plot(
            ax=ax,
            name=name,
            plot_chance_level=plot_chance_level,
            chance_level_kw=chance_level_kw,
        )
        if despine:
            _despine_matplotlib_axis(self.ax_)

        return self


class PredictionErrorDisplay(HelpDisplayMixin, PredictionErrorDisplay):
    pass
