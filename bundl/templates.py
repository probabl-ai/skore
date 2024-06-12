from jinja2

class TemplateRenderer:
    """We do a few things on top of Jinja2 here"""
    def __init__(self, datamander):
        self.datamander = datamander
    
    def clean_value(self, val): 
        return val.replace('/>', '').replace('"', '').replace("'", '')
        
    def insert_custom_ui(self, datamander, template):
        # For each registered element, check if it is there.
        for name, func in rendering_registry.items():
            element_of_interest = f'<{name}'
            start = template.find(element_of_interest)
            end = template[start:].find("/>")
            substr = template[start:start + end + 2]
            if substr:
                elems = [e.split('=') for e in substr.split(' ') if '=' in e]
                params = {k: clean_value(v) for k, v in elems}
                ui = func(datamander, **params)
                return template.replace(substr, ui)
        return template

    def render(self, datamander, template):
        final_template = self.insert_custom_ui(datamander, template)
        return 


def scatter_chart(datamander, title, data, x, y, **kwargs):
    # Grab the dataframe that is assumed to be stored in the datamander.
    dataf = pd.DataFrame(datamander.fetch()[data])

    # Render the altair chart internally
    c = alt.Chart(dataf).mark_circle(size=60).encode(
        x=x,
        y=y,
    ).interactive().properties(title=title)

    # Add the container width property
    json_blob = json.loads(c.to_json())
    json_blob['width'] = "container"
    
    return f'<vegachart style="width: 100%">{json.dumps(json_blob)}</vegachart>'
        
registry = {
    'scatter-chart': scatter_chart
}