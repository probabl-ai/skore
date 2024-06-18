import markdown
import pandas as pd 
import altair as alt
import json 
from jinja2 import Template


def scatter_chart(title, x, y, **kwargs):
    # Grab the dataframe that is assumed to be stored in the datamander.
    dataf = pd.DataFrame({'x': x, 'y': y})

    # Render the altair chart internally
    c = alt.Chart(dataf).mark_circle(size=60).encode(
        x='x',
        y='y',
    ).interactive().properties(title=title)

    # Add the container width property
    json_blob = json.loads(c.to_json())
    json_blob['width'] = "container"
    
    return c.to_html()


registry = {
    'scatter-chart': scatter_chart
}

class TemplateRenderer:
    """We do a few things on top of Jinja2 here"""
    def __init__(self, datamander):
        self.datamander = datamander
    
    def clean_value(self, val): 
        return val.replace('/>', '').replace('"', '').replace("'", '')
        
    def insert_custom_ui(self, template):
        # For each registered element, check if it is there.
        for name, func in registry.items():
            element_of_interest = f'<{name}'
            start = template.find(element_of_interest)
            end = template[start:].find("/>")
            substr = template[start:start + end + 2]
            if substr:
                elems = [e.split('=') for e in substr.split(' ') if '=' in e]
                params = {k: self.clean_value(v) for k, v in elems}
                for k, v in params.items():
                    if v.startswith('@mander'):
                        params[k] = self.datamander.get(v)
                ui = func(**params)
                return template.replace(substr, ui)
        return template

    def render(self, template):
        final_template = self.insert_custom_ui(template)
        return markdown.markdown(Template(final_template).render(**self.datamander.fetch()))
