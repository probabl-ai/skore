"""CLI for Skore."""

import argparse
from importlib.metadata import version

from skore.cli.color_format import ColorArgumentParser
from skore.project import open
from skore.project._create import _create
from skore.project._launch import _cleanup_potential_zombie_process, _launch
from skore.project._load import _load


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

    # launch the skore UI
    parser_launch = subparsers.add_parser("launch", help="Launch the web UI")
    parser_launch.add_argument(
        "project_name",
        help="the name or path of the project to open",
    )
    parser_launch.add_argument(
        "--keep-alive",
        action=argparse.BooleanOptionalAction,
        help=(
            "whether to keep the server alive once the main process finishes "
            "(default: %(default)s)"
        ),
        default="auto",
    )
    parser_launch.add_argument(
        "--port",
        type=int,
        help="the port at which to bind the UI server (default: %(default)s)",
        default=None,
    )
    parser_launch.add_argument(
        "--open-browser",
        action=argparse.BooleanOptionalAction,
        help=(
            "whether to automatically open a browser tab showing the web UI "
            "(default: %(default)s)"
        ),
        default=True,
    )
    parser_launch.add_argument(
        "--verbose",
        action="store_true",
        help="increase logging verbosity",
    )

    # cleanup potential zombie processes
    parser_cleanup = subparsers.add_parser(
        "cleanup", help="Clean up all UI running processes"
    )
    parser_cleanup.add_argument(
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
        "--keep-alive",
        action=argparse.BooleanOptionalAction,
        help=(
            "whether to keep the server alive once the main process finishes "
            "(default: %(default)s)"
        ),
        default="auto",
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

    parsed_args: argparse.Namespace = parser.parse_args(args)

    if parsed_args.subcommand == "create":
        _create(
            project_name=parsed_args.project_name,
            overwrite=parsed_args.overwrite,
            verbose=parsed_args.verbose,
        )
    elif parsed_args.subcommand == "launch":
        _launch(
            project=_load(project_name=parsed_args.project_name),
            keep_alive=parsed_args.keep_alive,
            port=parsed_args.port,
            open_browser=parsed_args.open_browser,
            verbose=parsed_args.verbose,
        )
    elif parsed_args.subcommand == "open":
        open(
            project_path=parsed_args.project_name,
            create=parsed_args.create,
            overwrite=parsed_args.overwrite,
            serve=parsed_args.serve,
            keep_alive=parsed_args.keep_alive,
            port=parsed_args.port,
            verbose=parsed_args.verbose,
        )
    elif parsed_args.subcommand == "cleanup":
        _cleanup_potential_zombie_process()
    else:
        parser.print_help()
