"""Entry point to Skore CLI."""

import sys

from skore.cli import cli

if __name__ == "__main__":
    cli(sys.argv[1:])
