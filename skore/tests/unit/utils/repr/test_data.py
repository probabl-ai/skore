"""Unit tests for helpers in ``skore._utils.repr.data``."""

from unittest.mock import patch
from urllib.parse import quote

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from skore._utils._testing import MockAccessor, MockDisplay, MockReport
from skore._utils.repr.data import (
    AccessorHelpData,
    DisplayHelpData,
    HelpSection,
    MethodHelp,
    ReportHelpData,
    _AccessorHelpDataMixin,
    _build_attribute_text_fragment,
    _DisplayHelpDataMixin,
    _get_attribute_type,
    _ReportHelpDataMixin,
    get_attribute_short_summary,
    get_documentation_url,
    get_method_short_summary,
    get_public_attributes,
    get_public_methods,
)


class _ClassWithNumpydocAttrs:
    """Class with numpydoc-style Attributes section.

    Parameters
    ----------
    param : int
        Parameter.

    Attributes
    ----------
    coef_ : ndarray (n_features,)
        Coefficients.
    """


class _ClassWithNoDocstring:
    pass


class _ClassWithDocstringMethod:
    """Helper for get_method_short_summary tests."""

    def documented(self):
        """First line of docstring."""
        pass

    def undocumented(self):
        pass


class _ReportWithExplicitMethods(MockReport, _ReportHelpDataMixin):
    """Report with explicit public, private, and class methods; used for help tests.

    Attributes
    ----------
    no_private : object
        Public attribute for tests.
    """

    def public_action(self):
        """A public method."""
        pass

    def _private_helper(self):
        pass

    @classmethod
    def class_factory(cls):
        """A class method; must be excluded from get_public_methods."""
        pass

    def _get_favorability_text(self, name: str) -> str | None:
        """Return favorability indicator for _build_help_data coverage."""
        if name == "public_action":
            return "(↗︎)"
        return None


class _ReportWithAccessor(_ReportWithExplicitMethods):
    """Report with _ACCESSOR_CONFIG and attached accessor for _build_help_data coverage."""

    _ACCESSOR_CONFIG = {"metrics": {"name": "metrics"}}

    def __init__(self, estimator, X_train=None, y_train=None, X_test=None, y_test=None):
        super().__init__(
            estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )
        self.metrics = _AccessorWithExplicitMethods(parent=self)


class _AccessorWithExplicitMethods(MockAccessor, _AccessorHelpDataMixin):
    """Accessor with explicit public, private, and class methods; used for help tests."""

    def fetch(self):
        """Fetch data."""
        pass

    def _internal(self):
        pass

    @classmethod
    def class_factory(cls):
        """A class method; must be excluded from get_public_methods."""
        pass

    def _get_help_title(self) -> str:
        return "Mock accessor"


class _EmptyAccessor(MockAccessor, _AccessorHelpDataMixin):
    """Accessor with no public methods; used for testing empty accessor exclusion."""

    def _internal(self):
        """Private method; should not be included."""
        pass

    def _get_help_title(self) -> str:
        return "Empty accessor"


class _ReportWithEmptyAccessor(_ReportWithExplicitMethods):
    """Report with _ACCESSOR_CONFIG and empty accessor for testing exclusion."""

    _ACCESSOR_CONFIG = {"empty": {"name": "empty"}}

    def __init__(self, estimator, X_train=None, y_train=None, X_test=None, y_test=None):
        super().__init__(
            estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )
        self.empty = _EmptyAccessor(parent=self)


class _DisplayWithExplicitMethods(MockDisplay, _DisplayHelpDataMixin):
    """Display with explicit private and class methods; used for help tests."""

    def _private_helper(self):
        pass

    @classmethod
    def class_factory(cls):
        """A class method; must be excluded from get_public_methods."""
        pass

    def _get_help_title(self) -> str:
        return "Mock display"


@pytest.mark.parametrize(
    "obj, attribute_name, expected",
    [
        (_ClassWithNumpydocAttrs(), "coef_", "ndarray (n_features,)"),
        (_ClassWithNumpydocAttrs(), "param", "int"),
        (_ClassWithNoDocstring(), "coef_", None),
        (_ClassWithNumpydocAttrs(), "other_attr", None),
    ],
)
def test_get_attribute_type(obj, attribute_name, expected):
    """_get_attribute_type returns the type for numpydoc entries, else None."""
    assert _get_attribute_type(obj, attribute_name) == expected


@pytest.mark.parametrize(
    "obj, attribute_name, expected",
    [
        (_ClassWithNumpydocAttrs(), "coef_", "ndarray (n_features,)"),
        (_ClassWithNumpydocAttrs(), "param", "int"),
        (_ClassWithNoDocstring(), "coef_", None),
        (_ClassWithNumpydocAttrs(), "other_attr", None),
    ],
)
def test_build_attribute_text_fragment(obj, attribute_name, expected):
    """_build_attribute_text_fragment returns encoded name,-type or encoded name."""
    got = _build_attribute_text_fragment(obj, attribute_name)
    if expected is not None:
        expected = f"{quote(attribute_name, safe='')},-{quote(expected, safe='')}"
    else:
        expected = quote(attribute_name, safe="")
    assert got == expected


@pytest.fixture
def report_with_methods():
    """Report with explicit public/private methods; used as main report fixture."""
    X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y = np.array([0, 1, 1, 0])
    estimator = LogisticRegression().fit(X, y)
    return _ReportWithExplicitMethods(estimator)


@pytest.fixture
def accessor_with_methods(report_with_methods):
    """Accessor with explicit public/private methods; used as main accessor fixture."""
    return _AccessorWithExplicitMethods(parent=report_with_methods)


@pytest.fixture
def display_with_methods():
    """Display with explicit private method; used as main display fixture."""
    return _DisplayWithExplicitMethods()


@pytest.fixture
def report_with_accessor():
    """Report with _ACCESSOR_CONFIG and metrics accessor for _build_help_data tests."""
    X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y = np.array([0, 1, 1, 0])
    estimator = LogisticRegression().fit(X, y)
    return _ReportWithAccessor(estimator)


def test_get_public_methods_display(display_with_methods):
    """get_public_methods returns public instance methods, excludes help, private, class."""
    methods = get_public_methods(display_with_methods)
    names = [n for n, _ in methods]
    for excluded in ("help", "_private_helper", "class_factory"):
        assert excluded not in names
    for expected in ("frame", "plot", "set_style"):
        assert expected in names
    assert names == sorted(names)


def test_get_public_methods_report_excludes_help(report_with_methods):
    """get_public_methods excludes help, private, and class methods, includes public ones."""
    methods = get_public_methods(report_with_methods)
    names = [n for n, _ in methods]
    for excluded in ("help", "_private_helper", "class_factory"):
        assert excluded not in names
    assert "public_action" in names


def test_get_public_methods_accessor_excludes_help(accessor_with_methods):
    """get_public_methods excludes help, private, and class methods, includes public ones."""
    methods = get_public_methods(accessor_with_methods)
    names = [n for n, _ in methods]
    for excluded in ("help", "_internal", "class_factory"):
        assert excluded not in names
    assert "fetch" in names


@pytest.mark.parametrize(
    "method_name, expected",
    [
        ("documented", "First line of docstring."),
        ("undocumented", "No description available"),
    ],
)
def test_get_method_short_summary(method_name, expected):
    """get_method_short_summary returns first docstring line or fallback."""
    obj = _ClassWithDocstringMethod()
    method = getattr(obj, method_name)
    assert get_method_short_summary(method) == expected


def test_get_public_attributes_report(report_with_methods):
    """get_public_attributes returns public non-callable attrs for report."""
    attrs = get_public_attributes(report_with_methods)
    for expected in ("no_private", "attr_without_description"):
        assert expected in attrs
    assert attrs == sorted(attrs)
    assert not any(n.startswith("_") for n in attrs)


def test_get_public_attributes_display(display_with_methods):
    """get_public_attributes returns public non-callable attrs for display."""
    attrs = get_public_attributes(display_with_methods)
    for excluded in ("plot", "frame"):
        assert excluded not in attrs
    assert attrs == sorted(attrs)


def test_get_public_attributes_accessor(accessor_with_methods):
    """get_public_attributes returns public non-callable attrs for accessor."""
    attrs = get_public_attributes(accessor_with_methods)
    assert "_parent" not in attrs
    assert attrs == sorted(attrs)


def test_get_attribute_short_summary_numpydoc(report_with_methods):
    """get_attribute_short_summary extracts description from numpydoc."""
    # Report has "no_private : object" -> "Public attribute for tests"
    summary = get_attribute_short_summary(report_with_methods, "no_private")
    assert "Public attribute" in summary and "tests" in summary

    # _ClassWithNumpydocAttrs has "coef_ : ndarray (...)" -> "Coefficients"
    obj = _ClassWithNumpydocAttrs()
    summary = get_attribute_short_summary(obj, "coef_")
    assert summary == "Coefficients"


def test_get_attribute_short_summary_not_found():
    """get_attribute_short_summary returns fallback when attr not in docstring."""
    obj = _ClassWithNumpydocAttrs()
    summary = get_attribute_short_summary(obj, "other_attr")
    assert summary == "No description available"


def test_get_attribute_short_summary_no_docstring():
    """get_attribute_short_summary returns fallback when no docstring."""
    obj = _ClassWithNoDocstring()
    summary = get_attribute_short_summary(obj, "coef_")
    assert summary == "No description available"


def test_get_documentation_url_class_only(report_with_methods, display_with_methods):
    """get_documentation_url returns base class URL with no accessor/method/attribute."""
    for obj in (report_with_methods, display_with_methods):
        url = get_documentation_url(obj=obj)
        assert url.startswith("https://docs.skore.probabl.ai/")
        assert "/reference/api/" in url
        assert "skore." in url and obj.__class__.__name__ in url
        assert url.endswith(".html")
        assert "#" not in url


def test_get_documentation_url_method_no_accessor(
    report_with_methods, display_with_methods
):
    """get_documentation_url appends #skore.Class.method when method set, no accessor."""
    for obj in (report_with_methods, display_with_methods):
        url = get_documentation_url(obj=obj, method_name="help")
        assert url.startswith("https://docs.skore.probabl.ai/")
        assert ".html#" in url
        assert "#skore." in url and "help" in url


def test_get_documentation_url_accessor(report_with_methods):
    """get_documentation_url includes accessor in path when accessor_name set."""
    url = get_documentation_url(obj=report_with_methods, accessor_name="data")
    cls = report_with_methods.__class__.__name__
    assert f"skore.{cls}.data.html" in url
    assert "#" not in url


def test_get_documentation_url_accessor_and_method(report_with_methods):
    """get_documentation_url includes accessor and method in path."""
    url = get_documentation_url(
        obj=report_with_methods, accessor_name="data", method_name="get_data"
    )
    cls = report_with_methods.__class__.__name__
    assert f"skore.{cls}.data.get_data.html" in url
    assert "#" not in url


def test_get_documentation_url_attribute():
    """get_documentation_url appends #:~:text= fragment when attribute_name set."""
    obj = _ClassWithNumpydocAttrs()
    url = get_documentation_url(obj=obj, attribute_name="coef_")
    assert url.startswith("https://docs.skore.probabl.ai/")
    assert ".html#:~:text=" in url
    assert quote("coef_", safe="") in url


@pytest.mark.parametrize(
    "mocked_version, expected_url_version",
    [
        ("0.0.1", "dev"),
        ("0.0.0.dev0", "dev"),
        ("0.1.0", "0.1"),
        ("1.2.3", "1.2"),
    ],
)
def test_get_documentation_url_version_branches(
    display_with_methods, mocked_version, expected_url_version
):
    """get_documentation_url uses \"dev\" for version < 0.1, else major.minor."""
    with patch("skore._utils.repr.data.version", return_value=mocked_version):
        url = get_documentation_url(obj=display_with_methods)
    assert url.startswith("https://docs.skore.probabl.ai/")
    assert f"docs.skore.probabl.ai/{expected_url_version}/reference/api/" in url


def test_accessor_build_help_data_output(accessor_with_methods):
    """_AccessorHelpDataMixin._build_help_data returns AccessorHelpData with expected shape."""
    data = accessor_with_methods._build_help_data()
    assert isinstance(data, AccessorHelpData)
    assert data.title == "Mock accessor"
    expected_root = accessor_with_methods._parent.__class__.__name__
    assert data.root_node == expected_root
    assert data.accessor_name == "mock_accessor"
    assert data.accessor_branch_id != ""
    assert len(data.methods) == 1
    m = data.methods[0]
    assert m.name == "fetch"
    assert m.parameters == "()"
    assert "Fetch" in m.description
    assert m.favorability is None
    assert m.doc_url.startswith("https://docs.skore.probabl.ai/")
    assert "mock_accessor" in m.doc_url and "fetch" in m.doc_url


def test_report_build_help_data_output(report_with_methods):
    """_ReportHelpDataMixin._build_help_data returns ReportHelpData with expected shape."""
    data = report_with_methods._build_help_data()
    assert isinstance(data, ReportHelpData)
    assert data.title == "Mock report"
    assert data.root_node == "_ReportWithExplicitMethods"
    assert data.class_name == "_ReportWithExplicitMethods"
    assert data.accessors == []
    assert len(data.base_methods) == 1
    m = data.base_methods[0]
    assert m.name == "public_action"
    assert m.parameters == "()"
    assert "public" in m.description.lower()
    assert m.favorability == "(↗︎)"
    assert m.doc_url.startswith("https://docs.skore.probabl.ai/")
    assert data.methods_section is not None
    assert isinstance(data.methods_section, HelpSection)
    assert data.attributes is not None
    assert data.attributes_section is not None


def test_report_build_help_data_output_with_accessors(report_with_accessor):
    """_ReportHelpDataMixin._build_help_data with _ACCESSOR_CONFIG builds accessor branches."""
    data = report_with_accessor._build_help_data()
    assert isinstance(data, ReportHelpData)
    assert len(data.accessors) == 1
    branch = data.accessors[0]
    assert branch.name == "metrics"
    assert branch.branch_id != ""
    assert len(branch.methods) == 1
    m = branch.methods[0]
    assert m.name == "fetch"
    assert m.parameters == "()"
    assert "Fetch" in m.description
    assert m.favorability is None
    assert m.doc_url.startswith("https://docs.skore.probabl.ai/")
    assert "metrics" in m.doc_url and "fetch" in m.doc_url


def test_report_build_help_data_output_excludes_empty_accessor():
    """_ReportHelpDataMixin._build_help_data excludes accessors with no methods."""
    X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y = np.array([0, 1, 1, 0])
    estimator = LogisticRegression().fit(X, y)
    report = _ReportWithEmptyAccessor(estimator)
    data = report._build_help_data()
    assert isinstance(data, ReportHelpData)
    assert len(data.accessors) == 0
    assert hasattr(report, "empty")
    assert report.empty is not None


def test_display_build_help_data_output(display_with_methods):
    """_DisplayHelpDataMixin._build_help_data returns DisplayHelpData with expected shape."""
    data = display_with_methods._build_help_data()
    assert isinstance(data, DisplayHelpData)
    assert data.title == "Mock display"
    assert data.root_node == "_DisplayWithExplicitMethods"
    assert data.class_name == "_DisplayWithExplicitMethods"
    assert data.attributes is None
    assert data.attributes_section is None
    assert data.methods_section is not None
    assert isinstance(data.methods_section, HelpSection)
    assert data.methods is not None
    assert len(data.methods) == 3
    names = {m.name for m in data.methods}
    assert names == {"frame", "plot", "set_style"}
    for m in data.methods:
        assert isinstance(m, MethodHelp)
        assert m.doc_url.startswith("https://docs.skore.probabl.ai/")
