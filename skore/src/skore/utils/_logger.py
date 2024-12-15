"""Module to handle the verbosity of a given logger."""

import logging
from functools import wraps

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


def with_logging(logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, verbose=False, **kwargs):
            if verbose:
                formatter = logging.Formatter("%(message)s")
                console_handler = Handler(markup=True)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if verbose:
                    logger.handlers.pop()

        return wrapper

    return decorator
