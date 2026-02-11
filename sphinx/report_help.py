"""
Sphinx extension for generating demo help HTML.

This module generates a demo help HTML file during the Sphinx build process
to showcase the skore reporting functionality.
"""

import sklearn
import skore
import skrub
from skrub.datasets import fetch_employee_salaries


def generate_help_demo():
    dataset = fetch_employee_salaries()
    df = dataset.X
    y = dataset.y

    report_ridge = skore.CrossValidationReport(
        skrub.tabular_pipeline(sklearn.linear_model.Ridge()), df, y
    )
    with open(
        "_templates/demo_report_help_generated.html", "w", encoding="utf-8"
    ) as f:
        f.write(report_ridge._create_help_html())

def setup(app):
    """Sphinx extension to generate demo help HTML."""

    generate_help_demo()
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
