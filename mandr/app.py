import json
from pathlib import Path

from flask import Flask, Response, request
from jinja2 import Template
from .infomander import InfoMander, VIEWS_KEY, LOGS_KEY, ARTIFACTS_KEY
from .templates import TemplateRenderer

from rich.console import Console


console = Console()

app = Flask(__name__)


def fetch_mander(*path):
    return InfoMander("/".join(path))


def render_views(*path):
    mander = fetch_mander(*path)
    view_nav_templ = read_template("partials/views.html")
    first_name = None
    if mander[VIEWS_KEY]:
        first_name = list(mander[VIEWS_KEY].items())[0][0]
    return view_nav_templ.render(
        views=list(mander[VIEWS_KEY].items()), first_name=first_name
    )


def render_info(*path):
    mander = fetch_mander(*path)
    return (
        '<pre class="text-xs">'
        + json.dumps(
            {k: str(v) for k, v in mander.fetch().items() if not k.startswith("_")},
            indent=2,
        )
        + "</pre>"
    )


def render_logs(*path):
    mander = fetch_mander(*path)
    view_nav_templ = read_template("partials/logs.html")
    return view_nav_templ.render(
        logs=list(mander[LOGS_KEY].items()), 
        first_name=list(mander[LOGS_KEY].items())[0][0]
    )


def render_artifacts(*path):
    mander = fetch_mander(*path)
    view_nav_templ = read_template('partials/artifacts.html')
    return view_nav_templ.render(artifacts=list(mander[ARTIFACTS_KEY].items()))

def read_template(path):
    p = Path(__file__).parent / "templates" / path
    return Template(p.read_text())


def render_top_nav(*args):
    nav_temp = read_template("partials/nav-top.html")
    path_pairs = []
    for i, p in enumerate(args):
        path_pairs.append(["/" + "/".join(args[: i + 1]), p])
    curr_file_path = Path(".datamander")
    for arg in args:
        curr_file_path = curr_file_path / arg
    glob = Path(curr_file_path).glob("*")
    links_out = [
        str(g).replace(".datamander", "")
        for g in glob
        if g.is_dir() and not g.parts[-1].startswith("_")
    ]
    console.log(f"{links_out=}")
    console.log(f"{path_pairs=}")
    return nav_temp.render(path_pairs=path_pairs, links_out=links_out)


def render_mid_nav(*args):
    nav_temp = read_template("partials/nav-mid.html")
    return nav_temp.render(path="/".join(args))


def render_mander(*args):
    p = Path(__file__).parent / "templates" / "page.html"
    t = Template(p.read_text())
    res = render_top_nav(*args)
    res += render_mid_nav(*args)
    return t.render(body=res)


@app.route("/", defaults={"path": ""}, methods=["GET", "POST"])
@app.route("/<path:path>", methods=["GET", "POST"])
def home(path):
    if "favicon" in path:
        return Response("", status=400)
    if len(path) == 0:
        return render_mander(*[])
    path_parts = path.split("/")
    console.log(f"{path_parts=}")
    if path_parts[0] == "info":
        return render_info(*path_parts[1:])
    if path_parts[0] == "view":
        return render_views(*path_parts[1:])
    if path_parts[0] == "logs":
        return render_logs(*path_parts[1:])
    if path_parts[0] == "artifacts":
        return render_artifacts(*path_parts[1:])
    if path_parts[0] == "sketchpad":
        return render_sketchpad(*path_parts[1:])
    if path_parts[0] == "render":
        return render_template(*path_parts[1:])
    return render_mander(*path.split("/"))


def render_sketchpad(*path):
    mander = fetch_mander(*path)
    children = [f"{m.path}" for m in mander.children()]
    return read_template("sketchpad.html").render(
        children=sorted(children), mander_path=mander.path
    )


def render_template(*path):
    mander = fetch_mander(*path)
    template_rendered = TemplateRenderer(mander)
    return template_rendered.render(request.form["template"])


if __name__ == "__main__":
    app.run(debug=True, reload=True)
