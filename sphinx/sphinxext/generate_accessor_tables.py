"""
Sphinx extension to automatically generate accessor method tables using Jinja templates.

This extension adds a config value `accessor_summary_classes` which should be a list
of class names to generate accessor summaries for. It automatically generates the
dropdown tables with accessor methods during the build process using a Jinja template.
"""

import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple

from jinja2 import Environment, FileSystemLoader
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


def get_accessor_data(
    class_name: str, accessor_config: Dict[str, Any], cls: type
) -> Dict[str, Any]:
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

    for accessor_key, accessor_info in accessor_config.items():
        accessor_name = accessor_info['name']
        methods = get_accessor_methods(cls, accessor_name)

        accessors[accessor_name] = {
            'methods': methods if methods else [('(no public methods)', '')]
        }

    return {
        'name': class_name,
        'accessors': accessors
    }


def generate_accessor_tables(app: Sphinx, config: Any) -> None:
    """
    Generate accessor table RST files using Jinja template during config-inited event.

    This function reads the `accessor_summary_classes` config value and generates
    RST snippets for each class that can be included in documentation.
    """
    classes_to_process = config.accessor_summary_classes

    if not classes_to_process:
        return

    logger.info("Generating accessor summary tables...")

    # Collect data for all classes
    classes_data = []

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

            # Get accessor data
            class_data = get_accessor_data(class_name, accessor_config, cls)
            classes_data.append(class_data)

            logger.info(f"Collected accessor data for {class_name}")

        except Exception as e:
            logger.error(f"Failed to process {class_path}: {e}", exc_info=True)

    # Render using Jinja template
    if classes_data:
        template_dir = Path(app.confdir) / '_templates'
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template('accessor_summary.rst')

        rst_content = template.render(classes=classes_data)

        output_path = Path(app.srcdir) / 'reference' / '_accessor_tables.rst'
        output_path.write_text(rst_content)
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
