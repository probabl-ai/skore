"""CLI for Skore."""

import argparse
from importlib.metadata import version

from skore.cli.color_format import ColorArgumentParser
from skore.project._open import open


def argumentparser():
    """Argument parser for the Skore CLI."""
    parser = ColorArgumentParser(
        prog="skore-ui",
        description="Launch the skore UI on a defined skore project.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('skore')}",
    )

    parser.add_argument(
        "project_name",
        help="the name or path of the project to be created or opened",
    )

    parser.add_argument(
        "--serve",
        action=argparse.BooleanOptionalAction,
        help=("whether to serve the project (default: %(default)s)"),
        default=True,
    )

    parser.add_argument(
        "--port",
        type=int,
        help="the port at which to bind the UI server (default: %(default)s)",
        default=22140,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="increase logging verbosity",
    )

    return parser


def cli(args: list[str]):
    """CLI for Skore."""
    parser = argumentparser()
    arguments = parser.parse_args(args)

    open(
        project_path=arguments.project_name,
        if_exists="load",
        serve=arguments.serve,
        keep_alive=True,
        port=arguments.port,
        verbose=arguments.verbose,
    )
