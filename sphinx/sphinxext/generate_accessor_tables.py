"""
Sphinx extension to automatically generate accessor method tables using Jinja templates.

This extension adds a config value `accessor_summary_classes` which should be a list
of class names to generate accessor summaries for. It automatically generates the
dropdown tables with accessor methods during the build process using a Jinja template.
"""

import inspect
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)


def get_accessor_methods(cls: type, accessor_name: str) -> list[tuple[str, str]]:
    """
    Get methods from an accessor by introspecting the accessor class.

    Args:
        cls: The report class (e.g., EstimatorReport)
        accessor_name: The accessor name (e.g., 'metrics')

    Returns:
        List of (method_name, description) tuples
    """
    accessor_cls = getattr(cls, accessor_name)

    if inspect.isclass(accessor_cls):
        # Already a class
        pass
    elif isinstance(accessor_cls, property):
        if accessor_cls.fget is None:
            return []

        sig = inspect.signature(accessor_cls.fget)

        if sig.return_annotation == inspect.Signature.empty:
            logger.debug(f"No return annotation for {cls.__name__}.{accessor_name}")
            return []

        accessor_cls = sig.return_annotation
    else:
        logger.debug(
            f"Unknown accessor type for {cls.__name__}.{accessor_name}: {type(accessor_cls)}"
        )
        return []

    methods = []
    for name in dir(accessor_cls):
        if name.startswith("_"):
            continue
        if name == "help":
            continue  # Skip help method

        attr = getattr(accessor_cls, name)

        if not callable(attr):
            continue

        doc = getattr(attr, "__doc__", "")
        if doc:
            first_line = doc.strip().split("\n")[0].strip()
            doc = first_line.rstrip(".")

        methods.append((name, doc))

    return sorted(methods)


def get_accessor_data(accessor_config: dict[str, Any], cls: type) -> dict[str, Any]:
    """
    Get accessor data for template rendering.

    Args:
        class_name: Short class name (e.g., 'EstimatorReport')
        accessor_config: The _ACCESSOR_CONFIG dictionary
        cls: The actual class object

    Returns:
        Dictionary with accessor data for template
    """
    accessors = {}

    for accessor_info in accessor_config.values():
        accessor_name = accessor_info["name"]
        methods = get_accessor_methods(cls, accessor_name) or [
            ("(no public methods)", "")
        ]
        accessors[accessor_name] = {"methods": methods}

    return {"name": cls.__name__, "accessors": accessors}


def generate_accessor_tables(app: Sphinx, config: Any) -> None:
    """
    Generate accessor table RST files using Jinja template during config-inited event.

    This function reads the `accessor_summary_classes` config value and generates
    RST snippets for each class that can be included in documentation.
    """
    classes_to_process = config.accessor_summary_classes

    if not classes_to_process:
        raise ValueError("accessor_summary_classes not found")

    logger.info("Generating accessor summary tables...")

    classes_data = []

    for class_path in classes_to_process:
        module_name, class_name = class_path.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)

        if not hasattr(cls, "_ACCESSOR_CONFIG"):
            raise ValueError(f"{cls} has no attribute '_ACCESSOR_CONFIG'.")

        accessor_config = cls._ACCESSOR_CONFIG

        class_data = get_accessor_data(accessor_config, cls)
        classes_data.append(class_data)

        logger.info(f"Collected accessor data for {class_name}")

    template_dir = Path(app.confdir) / "_templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("accessor_summary.rst")

    rst_content = template.render(classes=classes_data)

    (Path(app.srcdir) / "reference" / "api").mkdir(exist_ok=True)
    output_path = Path(app.srcdir) / "reference" / "api" / "accessor_tables.rst"
    output_path.write_text(rst_content)
    logger.info(f"Wrote accessor tables to {output_path}")


def setup(app: Sphinx) -> dict[str, Any]:
    """Setup the extension."""
    app.add_config_value("accessor_summary_classes", [], "html")
    app.connect("config-inited", generate_accessor_tables)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
