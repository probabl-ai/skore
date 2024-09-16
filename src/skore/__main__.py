"""Entry point to Skore CLI."""

import sys

from skore.cli.cli import cli

if __name__ == "__main__":
    import rich.traceback

    # Display error tracebacks with Rich
    rich.traceback.install(show_locals=True)

    cli(sys.argv[1:])
