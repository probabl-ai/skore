import numpy as np

from skore._sklearn.train_test_split import TrainTestSplit


class TestTrainTestSplit:
    """Tests for the TrainTestSplit splitter class."""

    def test_get_n_splits(self):
        splitter = TrainTestSplit()
        assert splitter.get_n_splits() == 1

    def test_split_yields_one_pair(self):
        X = np.arange(20).reshape(10, 2)
        splits = list(TrainTestSplit().split(X))
        assert len(splits) == 1
        train_idx, test_idx = splits[0]
        assert len(train_idx) + len(test_idx) == 10

    def test_defaults(self):
        splitter = TrainTestSplit()
        assert splitter.test_size == 0.2
        assert splitter.train_size is None
        assert splitter.random_state == 0
        assert splitter.shuffle is True
        assert splitter.stratify is None

    def test_parameters_stored(self):
        splitter = TrainTestSplit(
            test_size=0.4,
            train_size=0.5,
            random_state=7,
            shuffle=False,
            stratify=[0, 1],
        )
        assert splitter.test_size == 0.4
        assert splitter.train_size == 0.5
        assert splitter.random_state == 7
        assert splitter.shuffle is False
        assert splitter.stratify == [0, 1]
