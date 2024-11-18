"""Entry point to Skore CLI."""

import sys

from skore.cli.cli import cli


def main():
    """Entry point to Skore CLI."""
    import rich.traceback

    # Display error tracebacks with Rich
    rich.traceback.install(show_locals=True)

    sys.exit(cli(sys.argv[1:]))


if __name__ == "__main__":
    main()
