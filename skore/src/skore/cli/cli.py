"""CLI for Skore."""

import argparse
from importlib.metadata import version

from skore import Project
from skore.cli.color_format import ColorArgumentParser
from skore.cli.launch_dashboard import __launch


def cli(args: list[str]):
    """CLI for Skore."""
    parser = ColorArgumentParser(prog="skore")

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('skore')}"
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    parser_launch = subparsers.add_parser("launch", help="Launch the web UI")
    parser_launch.add_argument(
        "project_name",
        help="the name or path of the project to open",
    )
    parser_launch.add_argument(
        "--port",
        type=int,
        help="the port at which to bind the UI server (default: %(default)s)",
        default=22140,
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

    parser_quickstart = subparsers.add_parser(
        "quickstart", help='Create a "project.skore" file and start the UI'
    )
    parser_quickstart.add_argument(
        "project_name",
        nargs="?",
        help="the name or path of the project to create (default: %(default)s)",
        default="project",
    )
    parser_quickstart.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite an existing project with the same name",
    )
    parser_quickstart.add_argument(
        "--port",
        type=int,
        help="the port at which to bind the UI server (default: %(default)s)",
        default=22140,
    )
    parser_quickstart.add_argument(
        "--open-browser",
        action=argparse.BooleanOptionalAction,
        help=(
            "whether to automatically open a browser tab showing the web UI "
            "(default: %(default)s)"
        ),
        default=True,
    )
    parser_quickstart.add_argument(
        "--verbose",
        action="store_true",
        help="increase logging verbosity",
    )

    parsed_args: argparse.Namespace = parser.parse_args(args)

    if parsed_args.subcommand in ("create", "quickstart"):
        Project(
            path=parsed_args.project_name, create=True, overwrite=parsed_args.overwrite
        )

    if parsed_args.subcommand in ("launch", "quickstart"):
        __launch(
            project_name=parsed_args.project_name,
            port=parsed_args.port,
            open_browser=parsed_args.open_browser,
            verbose=parsed_args.verbose,
        )

    if parsed_args.subcommand not in ("create", "launch", "quickstart"):
        parser.print_help()
