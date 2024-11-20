from skore.sklearn.train_test_split.warning import (
    HighClassImbalanceTooFewExamplesWarning,
)


def test_check_high_class_imbalance_too_few_examples():
    y_test = [0] * 100
    y_labels = [0, 1]

    check_passed = HighClassImbalanceTooFewExamplesWarning.check(
        y_test=y_test,
        y_labels=y_labels,
        stratify=None,
        ml_task="multiclass-classification",
    )

    assert not check_passed
