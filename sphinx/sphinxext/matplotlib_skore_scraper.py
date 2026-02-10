from sphinx_gallery.scrapers import matplotlib_scraper


class matplotlib_skore_scraper:  # defining matplotlib scraper as a class not a function
    def __call__(self, *args, **kwargs):
        kwargs.setdefault("bbox_inches", "tight")
        return matplotlib_scraper(*args, **kwargs)


def setup(app):
    """Configure matplotlib scraper for sphinx-gallery."""
    app.config["sphinx_gallery_conf"] = {
        "image_scrapers": [matplotlib_skore_scraper()],
    }
