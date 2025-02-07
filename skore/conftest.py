import pytest
import sklearn
from sklearn.utils.fixes import parse_version


def pytest_collection_modifyitems(items):
    sklearn_version = parse_version(sklearn.__version__)
    min_required = parse_version("1.6")

    if sklearn_version < min_required:
        skip_marker = pytest.mark.skip(reason="Requires scikit-learn >= 1.6")
        for item in items:
            if isinstance(item, pytest.DoctestItem) and (
                "site-packages/skore" in str(item.path) or "src/skore" in str(item.path)
            ):
                item.add_marker(skip_marker)
