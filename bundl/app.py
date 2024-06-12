from flask import Flask, render_template
from pathlib import Path
from jinja2 import Template
from .infomander import InfoMander

app = Flask(__name__)

def fetch_mander(*path):
    InfoMander(*path)

def render_views():
    pass

def render_info(*path):
    mander = InfoMander(*path)
    pass

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
        print(args)
        path_pairs.append(['/' + '/'.join(args[:i+1]), p])
    print(path_pairs)
    return nav_temp.render(path_pairs=path_pairs)

def render_mander(*args):
    p = Path(__file__).parent / 'templates' / 'page.html'
    t = Template(p.read_text())
    res = render_top_nav(*args)
    res += t.render(breadcrumbs=args)
    return res

@app.route('/')
def home():
    return render_mander()

@app.route('/<path1>')
def home1(path1):
    print('this is hit')
    return render_mander(*[path1])

@app.route('/<path1>/<path2>')
def home2(path1, path2):
    return render_mander(*[path1, path2])

@app.route('/<path1>/<path2>/<path3>')
def home3(path1, path2, path3):
    return render_mander(*[path1, path2, path3])

@app.route('/<path1>/<path2>/<path3>/<path4>')
def home4(path1, path2, path3, path4):
    return render_mander(*[path1, path2, path3, path4])


if __name__ == '__main__':
    app.run(debug=True, reload=True)
