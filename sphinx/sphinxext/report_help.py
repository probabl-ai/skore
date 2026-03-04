"""
Sphinx extension for generating demo help HTML.

This module generates a demo help HTML file during the Sphinx build process
to showcase the skore reporting functionality.
"""

from typing import Any

from sphinx.application import Sphinx
import sklearn
import skore
import skrub


def generate_help_demo(_app: Sphinx, _config: Any) -> None:
    """Generate a demo of the Report help menu feature.

    Parameters are required by Sphinx but not used.
    """
    dataset = skrub.datasets.fetch_employee_salaries()
    X, y = dataset.X, dataset.y

    report_ridge = skore.CrossValidationReport(
        skrub.tabular_pipeline(sklearn.linear_model.Ridge()), X, y
    )
    (Path(app.srcdir) / "generated").mkdir(exist_ok=True)
    output_path = Path(app.srcdir) / "generated" / "demo_report_help_generated.html"
    output_path.write_text(report_ridge._create_help_html())


def setup(app):
    """Sphinx extension to generate demo help HTML."""
    app.connect("config-inited", generate_help_demo)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
