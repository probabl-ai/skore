"""CLI for Skore."""

import argparse
from importlib.metadata import version

from skore.cli.color_format import ColorArgumentParser
from skore.project import open
from skore.project._create import _create
from skore.project._launch import _kill_all_servers


def cli(args: list[str]):
    """CLI for Skore."""
    parser = ColorArgumentParser(prog="skore")

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('skore')}"
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    # create a skore project
    parser_create = subparsers.add_parser("create", help="Create a project")
    parser_create.add_argument(
        "project_name",
        nargs="?",
        help="the name or path of the project to create (default: %(default)s)",
        default="project",
    )
    parser_create.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite an existing project with the same name",
    )
    parser_create.add_argument(
        "--verbose",
        action="store_true",
        help="increase logging verbosity",
    )

    # open a skore project
    parser_open = subparsers.add_parser(
        "open", help='Create a "project.skore" file and start the UI'
    )
    parser_open.add_argument(
        "project_name",
        nargs="?",
        help="the name or path of the project to create (default: %(default)s)",
        default="project",
    )
    parser_open.add_argument(
        "--create",
        action=argparse.BooleanOptionalAction,
        help=("create a new project if it does not exist " "(default: %(default)s)"),
        default=True,
    )
    parser_open.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite an existing project with the same name",
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

    if parsed_args.subcommand == "create":
        _create(
            project_name=parsed_args.project_name,
            overwrite=parsed_args.overwrite,
            verbose=parsed_args.verbose,
        )
    elif parsed_args.subcommand == "open":
        open(
            project_path=parsed_args.project_name,
            create=parsed_args.create,
            overwrite=parsed_args.overwrite,
            serve=parsed_args.serve,
            keep_alive=True,
            port=parsed_args.port,
            verbose=parsed_args.verbose,
        )
    elif parsed_args.subcommand == "kill":
        _kill_all_servers(verbose=parsed_args.verbose)
    else:
        parser.print_help()
