"""
Sphinx extension to automatically generate accessor method tables using Jinja templates,
and to generate autosummary .inc files for accessor methods in report classes.

This extension adds two config values:

- ``accessor_summary_classes``: list of class paths to generate accessor summary
  dropdowns and individual accessor .inc files for, e.g.::

      accessor_summary_classes = [
          "skore.EstimatorReport",
          "skore.CrossValidationReport",
          "skore.ComparisonReport",
      ]

For each accessor in each report class, a file is generated at
``reference/api/<ClassName>.<accessor_name>.inc`` containing the autosummary
directive with fully-qualified paths. The report RST files should include
these with ``.. include::``.
"""

import inspect
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)


def get_doc_first_line(obj):
    doc = getattr(obj, "__doc__", "")
    if doc:
        first_line = doc.strip().split("\n")[0].strip()
        doc = first_line.rstrip(".")
    return doc


def get_accessor_methods(cls: type, accessor_name: str) -> list[tuple[str, str]]:
    """Get methods from an accessor by introspecting the accessor class.

    Parameters
    ----------
    cls : type
        The report class (e.g., EstimatorReport).
    accessor_name : str
        The accessor name (e.g., 'metrics').

    Returns
    -------
    list of tuple of (str, str)
        List of (method_name, description) tuples.
    """
    accessor_cls = getattr(cls, accessor_name)
    if inspect.isclass(accessor_cls):
        pass
    elif isinstance(accessor_cls, property):
        if accessor_cls.fget is None:
            return []
        sig = inspect.signature(accessor_cls.fget)
        if sig.return_annotation == inspect.Signature.empty:
            logger.warning(
                f"No return annotation for {cls.__name__}.{accessor_name}, "
                "accessor will be skipped"
            )
            return []
        accessor_cls = sig.return_annotation
    else:
        logger.debug(
            f"Unknown accessor type for {cls.__name__}.{accessor_name}: "
            f"{type(accessor_cls)}"
        )
        return []

    methods = []
    for name in dir(accessor_cls):
        if name.startswith("_"):
            continue
        attr = getattr(accessor_cls, name)
        if not callable(attr):
            continue
        doc = get_doc_first_line(attr)
        methods.append((name, doc))
    return sorted(methods)


def get_accessor_data(cls: type) -> dict[str, Any]:
    """Get accessor data for template rendering.

    Parameters
    ----------
    cls : type
        The class object.

    Returns
    -------
    dict
        Dictionary with accessor data for template.
    """
    if not hasattr(cls, "_ACCESSOR_CONFIG"):
        raise ValueError(f"{cls} has no attribute '_ACCESSOR_CONFIG'.")
    accessor_config: dict = cls._ACCESSOR_CONFIG
    accessors = {}
    for accessor_info in accessor_config.values():
        accessor_name = accessor_info["name"]
        methods = get_accessor_methods(cls, accessor_name) or [
            ("(no public methods)", "")
        ]
        accessors[accessor_name] = {"methods": methods}
    return {
        "name": cls.__name__,
        "doc": get_doc_first_line(cls),
        "accessors": accessors,
    }


def generate_accessor_tables(app: Sphinx, config: Any) -> None:
    """Generate accessor table RST and individual accessor .inc files.

    Parameters
    ----------
    app : Sphinx
        The Sphinx application object.
    config : Any
        The Sphinx configuration object.
    """
    classes_to_process = config.accessor_summary_classes
    if not classes_to_process:
        return

    logger.info("Generating accessor summary tables...")
    classes_data = []
    for class_path in classes_to_process:
        module_name, class_name = class_path.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        class_data = get_accessor_data(cls)
        classes_data.append((class_path, class_data))
        logger.info(f"Collected accessor data for {class_name}")

    template_dir = Path(app.confdir) / "_templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("accessor_summary.rst")
    rst_content = template.render(classes=[cd for _, cd in classes_data])

    output_dir = Path(app.srcdir) / "reference" / "api"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "accessor_tables.rst"
    output_path.write_text(rst_content)
    logger.info(f"Wrote accessor tables to {output_path}")

    # Generate individual .inc files for each accessor
    logger.info("Generating individual accessor .inc files...")
    for class_path, class_data in classes_data:
        class_name = class_data["name"]

        for accessor_name, data in class_data["accessors"].items():
            methods = data["methods"]
            if not methods or (
                len(methods) == 1 and methods[0][0] == "(no public methods)"
            ):
                continue

            # Put help and summarize first (navigation methods), then rest alphabetically
            priority = [m for m in methods if m[0] in ("help", "summarize")]
            rest = [m for m in methods if m[0] not in ("help", "summarize")]
            ordered = priority + rest

            lines = [
                ".. currentmodule:: skore",
                "",
                ".. autosummary::",
                "    :toctree: ../api/",
                "    :template: autosummary/accessor_method.rst",
                "",
            ]
            for method_name, _ in ordered:
                lines.append(f"    {class_name}.{accessor_name}.{method_name}")
            lines.append("")

            filename = f"{class_name}.{accessor_name}.inc"
            file_path = output_dir / filename

            content = "\n".join(lines)
            if not file_path.exists() or file_path.read_text() != content:
                file_path.write_text(content)
                logger.info(f"Updated {filename}")
            else:
                logger.debug(f"No changes for {filename}")


def setup(app: Sphinx) -> dict[str, Any]:
    """Setup the extension.

    Parameters
    ----------
    app : Sphinx
        The Sphinx application object.

    Returns
    -------
    dict
        Extension metadata including version and parallel safety flags.
    """
    app.add_config_value("accessor_summary_classes", [], "html")
    app.connect("config-inited", generate_accessor_tables)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
