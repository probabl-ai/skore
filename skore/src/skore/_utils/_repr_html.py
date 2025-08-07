class ReprHTMLMixin:
    """Mixin to handle consistently the HTML representation.

    When inheriting from this class, you need to define an attribute `_html_repr`
    which is a callable that returns the HTML representation to be shown.
    """

    @property
    def _repr_html_(self):
        return self._repr_html_inner

    def _repr_html_inner(self):
        return self._html_repr()

    def _repr_mimebundle_(self, **kwargs):
        return {"text/plain": repr(self), "text/html": self._html_repr()}
