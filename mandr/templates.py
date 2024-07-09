"""Contains all the logic to render templates and custom UI elements."""

import json

import altair as alt
import markdown
import pandas as pd
from jinja2 import Template
from sklearn.utils._estimator_html_repr import estimator_html_repr


def sklearn_model_repr(pipeline):
    """Render a HTML representation of the scikit-learn pipeline."""
    return estimator_html_repr(pipeline)


def scatter_chart(title, x, y, **kwargs):
    """
    Render a scatter chart using Altair.
    
    Parameters
    ----------
    title : str
        The title of the chart.
    x : list
        The x values.
    y : list
        The y values.
    """
    # Grab the dataframe that is assumed to be stored in the datamander.
    dataf = pd.DataFrame({"x": x, "y": y})

    # Render the altair chart internally
    c = (
        alt.Chart(dataf)
        .mark_circle(size=60)
        .encode(
            x="x",
            y="y",
        )
        .interactive()
        .properties(title=title)
    )

    # Add the container width property
    json_blob = json.loads(c.to_json())
    json_blob["width"] = "container"

    return c.to_html()


registry = {"scatter-chart": scatter_chart, "sklearn-model-repr": sklearn_model_repr}


class TemplateRenderer:
    """Class that can render templates and custom UI elements."""

    def __init__(self, mander):
        """Initialize the renderer with the mander."""
        self.mander = mander

    def clean_value(self, val):
        """Clean the value by removing quotes and />."""
        return val.replace("/>", "").replace('"', "").replace("'", "")

    def insert_custom_ui(self, template):
        """
        Insert the custom UI elements into the template.
        
        Parameters
        ----------
        template : str
            The template to insert the custom UI elements into.
        """
        # For each registered element, check if it is there.
        for name, func in registry.items():
            element_of_interest = f"<{name}"
            start = template.find(element_of_interest)
            end = template[start:].find("/>")
            substr = template[start : start + end + 2]
            if substr:
                elems = [e.split("=") for e in substr.split(" ") if "=" in e]
                params = {k: self.clean_value(v) for k, v in elems}
                for k, v in params.items():
                    if v.startswith("@mander"):
                        params[k] = self.mander.get(v)
                ui = func(**params)
                template = template.replace(substr, ui)
        return template

    def render(self, template):
        """
        Render the template with the mander.
        
        Parameters
        ----------
        template : str
            The template to render.
        """
        final_template = self.insert_custom_ui(template)
        res = markdown.markdown(Template(final_template).render(**self.mander.fetch()))
        return res
