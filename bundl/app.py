from flask import Flask, render_template
from pathlib import Path
from jinja2 import Template
from .infomander import InfoMander

app = Flask(__name__)

def fetch_mander(*path):
    InfoMander(*path)

def render_views(name, path):
    pass

def render_info(*path):
    mander = InfoMander(*path)
    return mander.fetch()

def render_logs():
    pass

def render_artifacts():
    pass

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
    if path_parts[0] == 'views':
        return 'info fetches!'
    if path_parts[0] == 'logs':
        return 'info fetches!'
    if path_parts[0] == 'artifacts':
        return 'info fetches!'
    return render_mander(*path.split('/'))

if __name__ == '__main__':
    app.run(debug=True, reload=True)
