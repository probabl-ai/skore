from sphinx_gallery.scrapers import matplotlib_scraper


class matplotlib_skore_scraper: # defining matplotlib scraper as a class not a function
    def __call__(self, *args, **kwargs):
        kwargs.setdefault("bbox_inches", "tight")
        return matplotlib_scraper(*args, **kwargs)
