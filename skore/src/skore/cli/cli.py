"""CLI for Skore."""

import argparse
from importlib.metadata import version

from skore.cli.color_format import ColorArgumentParser
from skore.project import open
from skore.project._launch import _kill_all_servers


def cli(args: list[str]):
    """CLI for Skore."""
    parser = ColorArgumentParser(prog="skore-ui")

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('skore')}"
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    # open a skore project
    parser_open = subparsers.add_parser(
        "open", help="Open a skore project and start the UI"
    )
    parser_open.add_argument(
        "project_path",
        nargs="?",
        help="the name or path of the project to be opened",
    )
    parser_open.add_argument(
        "--serve",
        action=argparse.BooleanOptionalAction,
        help=("whether to serve the project (default: %(default)s)"),
        default=True,
    )
    parser_open.add_argument(
        "--port",
        type=int,
        help="the port at which to bind the UI server (default: %(default)s)",
        default=None,
    )
    parser_open.add_argument(
        "--verbose",
        action="store_true",
        help="increase logging verbosity",
    )

    # kill all UI servers
    parser_kill = subparsers.add_parser("kill", help="Kill all UI servers")
    parser_kill.add_argument(
        "--verbose",
        action="store_true",
        help="increase logging verbosity",
    )

    parsed_args: argparse.Namespace = parser.parse_args(args)

    if parsed_args.subcommand == "open":
        open(
            project_path=parsed_args.project_path,
            if_exists="load",
            serve=parsed_args.serve,
            keep_alive=True,
            port=parsed_args.port,
            verbose=parsed_args.verbose,
        )
    elif parsed_args.subcommand == "kill":
        _kill_all_servers(verbose=parsed_args.verbose)
    else:
        parser.print_help()
