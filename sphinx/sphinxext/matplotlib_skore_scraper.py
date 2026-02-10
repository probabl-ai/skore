from sphinx_gallery.scrapers import matplotlib_scraper


class matplotlib_skore_scraper:  # defining matplotlib scraper as a class not a function
    def __call__(self, *args, **kwargs):
        kwargs.setdefault("bbox_inches", "tight")
        return matplotlib_scraper(*args, **kwargs)


def setup(app):
    if "sphinx_gallery_conf" not in app.config:
        app.config["sphinx_gallery_conf"] = {}

    # Append to existing image_scrapers rather than replacing them
    if "image_scrapers" not in app.config["sphinx_gallery_conf"]:
        app.config["sphinx_gallery_conf"]["image_scrapers"] = []

    app.config["sphinx_gallery_conf"]["image_scrapers"].append(matplotlib_skore_scraper())
