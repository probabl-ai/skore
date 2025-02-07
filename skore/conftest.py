import sklearn
from sklearn.utils.fixes import parse_version

def pytest_collect_file(parent, path):
    if path.strpath.startswith('src/'):
        sklearn_version = parse_version(sklearn.__version__)
        required_version = parse_version('1.6')

        if sklearn_version < required_version:
            return None

    return None
