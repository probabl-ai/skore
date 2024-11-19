from skore.sklearn.train_test_split.warning import StratifyWarning


def test_check_stratify():
    check_passed = StratifyWarning.check(stratify=[0] * 100)
    assert not check_passed


def test_check_stratify_passes():
    check_passed = StratifyWarning.check(stratify=None)
    assert check_passed
