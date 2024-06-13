from flask import Flask, render_template
from pathlib import Path
import orjson
import json
from jinja2 import Template
from .infomander import InfoMander

app = Flask(__name__)

def fetch_mander(*path):
    InfoMander(*path)

def render_views(*path):
    mander = InfoMander(*path)
    view_nav_templ = read_template('partials/views.html')
    return view_nav_templ.render(
        views=list(mander['_templates'].items()), 
        first_name=list(mander['_templates'].items())[0][0]
    )

def render_info(*path):
    mander = InfoMander(*path)
    print(mander.fetch().keys())
    return '<pre class="text-xs">' + json.dumps({k: str(v) for k, v in mander.fetch().items() if not k.startswith("_")}, indent=2) + '</pre>'

def render_logs(*path):
    mander = InfoMander(*path)
    view_nav_templ = read_template('partials/logs.html')
    return view_nav_templ.render(
        logs=list(mander['_logs'].items()), 
        first_name=list(mander['_logs'].items())[0][0]
    )

def render_artifacts(*path):
    mander = InfoMander(*path)
    view_nav_templ = read_template('partials/artifacts.html')
    return view_nav_templ.render(artifacts=list(mander['_artifacts'].items()))

def read_template(path):
    p = Path(__file__).parent / 'templates' / path
    return Template(p.read_text())

def render_top_nav(*args):
    nav_temp = read_template('partials/nav-top.html')
    path_pairs = []
    for i, p in enumerate(args):
        path_pairs.append(['/' + '/'.join(args[:i+1]), p])
    curr_file_path = Path('.datamander')
    for arg in args:
        curr_file_path = curr_file_path / arg
    glob = Path(curr_file_path).glob("*")
    links_out = [str(g).replace('.datamander', '') for g in glob if g.is_dir()]
    return nav_temp.render(path_pairs=path_pairs, links_out=links_out)

def render_mid_nav(*args):
    nav_temp = read_template('partials/nav-mid.html')
    return nav_temp.render(path='/'.join(args))

def render_mander(*args):
    p = Path(__file__).parent / 'templates' / 'page.html'
    t = Template(p.read_text())
    res = render_top_nav(*args)
    res += render_mid_nav(*args)
    return t.render(body=res)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def home(path):
    if len(path) == 0:
        return render_mander(*[])
    path_parts = path.split('/')
    print(path_parts)
    if path_parts[0] == 'info':
        return render_info(*path_parts[1:])
    if path_parts[0] == 'view':
        return render_views(*path_parts[1:])
    if path_parts[0] == 'logs':
        return render_logs(*path_parts[1:])
    if path_parts[0] == 'artifacts':
        return render_artifacts(*path_parts[1:])
    return render_mander(*path.split('/'))

if __name__ == '__main__':
    app.run(debug=True, reload=True)
