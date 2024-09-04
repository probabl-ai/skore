"""Entry point to Mandr CLI."""

import sys

from mandr.cli import cli

if __name__ == "__main__":
    cli(sys.argv[1:])
