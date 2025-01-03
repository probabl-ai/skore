"""Implement skore CLI."""

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Default to no output
logger.setLevel(logging.INFO)
logger.propagate = False
