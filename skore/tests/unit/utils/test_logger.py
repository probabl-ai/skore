import logging

import pytest
from skore._utils._logger import Handler, logger_context


def test_logger_context_verbose():
    """Test that logger_context properly adds and removes handler when verbose=True."""
    logger = logging.getLogger("test_logger")
    initial_handlers = list(logger.handlers)

    with logger_context(logger, verbose=True):
        current_handlers = list(logger.handlers)
        assert len(current_handlers) == len(initial_handlers) + 1
        added_handler = current_handlers[-1]
        assert isinstance(added_handler, Handler)
        assert added_handler.formatter._fmt == "%(message)s"
        assert added_handler.markup is True

    final_handlers = list(logger.handlers)
    assert final_handlers == initial_handlers


def test_logger_context_non_verbose():
    """Test that logger_context doesn't modify handlers when verbose=False."""
    logger = logging.getLogger("test_logger")
    initial_handlers = list(logger.handlers)

    with logger_context(logger, verbose=False):
        assert list(logger.handlers) == initial_handlers

    assert list(logger.handlers) == initial_handlers


def test_logger_context_exception():
    """Test that logger_context removes handler even if an exception occurs."""
    logger = logging.getLogger("test_logger")
    initial_handlers = list(logger.handlers)

    with pytest.raises(ValueError), logger_context(logger, verbose=True):
        raise ValueError("Test exception")

    assert list(logger.handlers) == initial_handlers
