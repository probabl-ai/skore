from sphinx_gallery.scrapers import matplotlib_scraper


def matplotlib_skore_scraper(*args, **kwargs):
    return matplotlib_scraper(*args, bbox_inches="tight", **kwargs)
