from flask import Flask, Response, request
from pathlib import Path
import json
from jinja2 import Template
from .infomander import InfoMander, VIEWS_KEY
from rich.console import Console

console = Console()

app = Flask(__name__)
    

def fetch_mander(*path):
    return InfoMander('/'.join(path))


def render_views(*path):
    mander = fetch_mander(*path)
    view_nav_templ = read_template('partials/views.html')
    first_name = None
    if mander[VIEWS_KEY]:
        first_name = list(mander[VIEWS_KEY].items())[0][0]
    return view_nav_templ.render(
        views=list(mander[VIEWS_KEY].items()), 
        first_name=first_name
    )


def render_info(*path):
    mander = fetch_mander(*path)
    return '<pre class="text-xs">' + json.dumps({k: str(v) for k, v in mander.fetch().items() if not k.startswith("_")}, indent=2) + '</pre>'


def render_logs(*path):
    mander = fetch_mander(*path)
    view_nav_templ = read_template('partials/logs.html')
    return view_nav_templ.render(
        logs=list(mander['_logs'].items()), 
        first_name=list(mander['_logs'].items())[0][0]
    )


def render_artifacts(*path):
    mander = fetch_mander(*path)
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
    links_out = [str(g).replace('.datamander', '') for g in glob if g.is_dir() and not g.parts[-1].startswith('_')]
    console.log(f'{links_out=}')
    console.log(f'{path_pairs=}')
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
    if 'favicon' in path:
        return Response('', status=400)
    if len(path) == 0:
        return render_mander(*[])
    path_parts = path.split('/')
    console.log(f'{path_parts=}')
    if path_parts[0] == 'info':
        return render_info(*path_parts[1:])
    if path_parts[0] == 'view':
        return render_views(*path_parts[1:])
    if path_parts[0] == 'logs':
        return render_logs(*path_parts[1:])
    if path_parts[0] == 'artifacts':
        return render_artifacts(*path_parts[1:])
    return render_mander(*path.split('/'))


@app.route('/sketchpad')
def sketchpad():
    return read_template('sketchpad.html').render()


@app.post('/autocomplete')
def autocomplete():
    last_path = Path(request.form['path'])
    if (Path('.datamander') / last_path).exists():
        entry_path = Path(f'.datamander/{last_path}')
    else:
        entry_path = Path(f'.datamander/{last_path.parent}')
    console.log(f'{entry_path=} {last_path=}')
    if entry_path.exists() and entry_path.parts[0] == '.datamander' and entry_path.is_dir():
        paths = entry_path.iterdir()
    else:
        paths = Path('.datamander').iterdir()
    print(request.form)
    return ''.join(f'<p>{p}</p>' for p in paths if p.is_dir() and not p.parts[-1].startswith('.'))


@app.post('/render')
def render():
    print(request.form)
    return 'this is the rendered template/mander combo'


if __name__ == '__main__':
    app.run(debug=True, reload=True)
