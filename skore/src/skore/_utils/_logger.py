"""Module to handle the verbosity of a given logger."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from rich.logging import LogRecord, RichHandler, Text


class Handler(RichHandler):
    """A logging handler that renders output with Rich."""

    def get_level_text(self, record: LogRecord) -> Text:
        """Get the logger name and levelname from the record."""
        levelname = record.levelname
        name = record.name
        txt = f"<Logger {name} ({levelname})>"

        return Text.styled(
            txt.ljust(8),
            f"logging.level.{levelname.lower()}",
        )


@contextmanager
def logger_context(logger: logging.Logger, verbose: bool = False) -> Iterator[None]:
    """Context manager for temporarily adding a Rich handler to a logger."""
    handler = None
    try:
        if verbose:
            formatter = logging.Formatter("%(message)s")
            handler = Handler(markup=True)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        yield
    finally:
        if verbose and handler:
            logger.removeHandler(handler)
