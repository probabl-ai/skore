"""
Sphinx extension to automatically generate accessor method tables.

This extension adds a config value `accessor_summary_classes` which should be a list
of class names to generate accessor summaries for. It automatically generates the
dropdown tables with accessor methods during the build process.
"""

import inspect
from typing import Any, Dict, List, Tuple

from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)


def get_accessor_methods(cls: type, accessor_name: str) -> List[Tuple[str, str]]:
    """
    Get methods from an accessor by introspecting the accessor class.

    Args:
        cls: The report class (e.g., EstimatorReport)
        accessor_name: The accessor name (e.g., 'metrics')

    Returns:
        List of (method_name, description) tuples
    """
    try:
        # Get the accessor descriptor/class
        accessor_cls = getattr(cls, accessor_name)

        # It could be a property or a descriptor class
        if isinstance(accessor_cls, property):
            # It's a property, get the return type from fget
            fget = accessor_cls.fget
            if fget is None:
                return []

            sig = inspect.signature(fget)
            return_annotation = sig.return_annotation

            if return_annotation == inspect.Signature.empty:
                logger.debug(f"No return annotation for {cls.__name__}.{accessor_name}")
                return []

            accessor_cls = return_annotation
        elif inspect.isclass(accessor_cls):
            # It's already a class (descriptor pattern)
            pass
        else:
            logger.debug(f"Unknown accessor type for {cls.__name__}.{accessor_name}: {type(accessor_cls)}")
            return []

        # Get public methods from the accessor class
        methods = []
        for name in dir(accessor_cls):
            if name.startswith('_'):
                continue
            if name == 'help':
                continue  # Skip help method

            try:
                attr = getattr(accessor_cls, name)
                if not callable(attr):
                    continue

                # Get the docstring
                doc = getattr(attr, '__doc__', '')
                if doc:
                    # Extract first line, clean it up
                    first_line = doc.strip().split('\n')[0].strip()
                    # Remove any leading/trailing periods or extra whitespace
                    doc = first_line.rstrip('.')
                else:
                    doc = ''

                methods.append((name, doc))
            except Exception as e:
                logger.debug(f"Error introspecting {name}: {e}")
                continue

        return sorted(methods)

    except Exception as e:
        logger.warning(f"Error getting methods for {cls.__name__}.{accessor_name}: {e}")
        return []


def generate_accessor_table_rst(
    class_name: str, accessor_config: Dict[str, Any], cls: type
) -> str:
    """
    Generate RST content for accessor tables.

    Args:
        class_name: Short class name (e.g., 'EstimatorReport')
        accessor_config: The _ACCESSOR_CONFIG dictionary
        cls: The actual class object

    Returns:
        RST content as a string
    """
    rst_lines = []
    rst_lines.append(f".. dropdown:: {class_name} accessors")
    rst_lines.append("   :icon: chevron-down")
    rst_lines.append("")

    for accessor_key, accessor_info in accessor_config.items():
        accessor_name = accessor_info['name']
        accessor_full_name = f"{class_name}.{accessor_name}"

        rst_lines.append(f"   **{accessor_name.capitalize()}**: :class:`{accessor_full_name}`")
        rst_lines.append("")
        rst_lines.append("   .. list-table::")
        rst_lines.append("      :widths: 30 70")
        rst_lines.append("")

        # Get accessor methods
        methods = get_accessor_methods(cls, accessor_name)

        if methods:
            for method_name, method_doc in methods:
                rst_lines.append(f"      * - :func:`~{accessor_full_name}.{method_name}`")
                rst_lines.append(f"        - {method_doc}")
        else:
            rst_lines.append("      * - (no public methods)")
            rst_lines.append("        -")

        rst_lines.append("")

    return '\n'.join(rst_lines)


def generate_accessor_tables(app: Sphinx, config: Any) -> None:
    """
    Generate accessor table RST files during config-inited event.

    This function reads the `accessor_summary_classes` config value and generates
    RST snippets for each class that can be included in documentation.
    """
    from pathlib import Path

    classes_to_process = config.accessor_summary_classes

    if not classes_to_process:
        return

    logger.info("Generating accessor summary tables...")

    # Collect all RST content
    all_rst = []

    for class_path in classes_to_process:
        try:
            # Import the class
            module_name, class_name = class_path.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)

            # Check if class has _ACCESSOR_CONFIG
            if not hasattr(cls, '_ACCESSOR_CONFIG'):
                logger.warning(f"{class_path} does not have _ACCESSOR_CONFIG")
                continue

            accessor_config = cls._ACCESSOR_CONFIG

            # Generate RST content
            rst_content = generate_accessor_table_rst(class_name, accessor_config, cls)
            all_rst.append(rst_content)
            all_rst.append("")  # Add blank line between dropdowns

            logger.info(f"Generated accessor table for {class_name}")

        except Exception as e:
            logger.error(f"Failed to process {class_path}: {e}", exc_info=True)

    # Write all content to a single file
    if all_rst:
        output_path = Path(app.srcdir) / 'reference' / '_accessor_tables.rst'
        output_path.write_text('\n'.join(all_rst))
        logger.info(f"Wrote accessor tables to {output_path}")


def setup(app: Sphinx) -> Dict[str, Any]:
    """Setup the extension."""
    app.add_config_value('accessor_summary_classes', [], 'html')
    app.connect('config-inited', generate_accessor_tables)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
