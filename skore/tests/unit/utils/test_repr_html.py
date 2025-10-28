import pytest

from skore._utils._repr_html import ReprHTMLMixin


@pytest.fixture
def class_with_repr_html():
    class ClassWithReprHTML(ReprHTMLMixin):
        def _html_repr(self):
            return "html_repr"

    return ClassWithReprHTML()


def test_repr_mimebundle(class_with_repr_html):
    output = class_with_repr_html._repr_mimebundle_()
    assert "text/plain" in output
    assert "text/html" in output


def test_repr_html_wraps(class_with_repr_html):
    output = class_with_repr_html._repr_html_()
    assert "html_repr" in output
