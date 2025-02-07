import pytest
import sklearn
from sklearn.utils.fixes import parse_version


def pytest_collection_modifyitems(items):
    sklearn_version = parse_version(sklearn.__version__)
    min_required = parse_version("1.6")

    if sklearn_version < min_required:
        skip_marker = pytest.mark.skip(reason="Requires scikit-learn >= 1.6")
        for item in items:
            if isinstance(item, pytest.DoctestItem):
                path_str = str(getattr(item, 'path', '') or getattr(item, 'fspath', ''))
                if "site-packages/skore" in path_str or "src/skore" in path_str:
                    item.add_marker(skip_marker)
