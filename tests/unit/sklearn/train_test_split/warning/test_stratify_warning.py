from skore.sklearn.train_test_split.warning import StratifyWarning


def test_check_stratify():
    warning = StratifyWarning.check(stratify=[0] * 100)
    assert warning == StratifyWarning.MSG


def test_check_stratify_passes():
    warning = StratifyWarning.check(stratify=None)
    assert warning is None
