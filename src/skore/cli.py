"""CLI for Skore."""

import argparse
import pathlib
from importlib.metadata import version

from skore.create_project import create_project
from skore.dashboard.dashboard import __launch
from skore.quickstart_command import __quickstart


def cli(args: list[str]):
    """CLI for Skore."""
    parser = argparse.ArgumentParser(prog="skore")

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version("skore")}"
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    parser_launch = subparsers.add_parser("launch", help="Launch the dashboard")
    parser_launch.add_argument(
        "project_name",
        nargs="?",
        help="the name of the project to open (default: %(default)s)",
        default="project",
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
            "whether to automatically open a browser tab showing the dashboard "
            "(default: %(default)s)"
        ),
        default=True,
    )

    parser_create = subparsers.add_parser("create", help="Create a project")
    parser_create.add_argument(
        "project_name",
        nargs="?",
        help="the name of the project (default: %(default)s)",
        default="project",
    )
    parser_create.add_argument(
        "--working-dir",
        type=pathlib.Path,
        help=(
            "the directory relative to which the project name will be interpreted; "
            "default is the current working directory (mostly used for testing)"
        ),
        default=None,
    )

    subparsers.add_parser(
        "quickstart", help='Create a "project.skore" file and start the dashboard'
    )

    parsed_args: argparse.Namespace = parser.parse_args(args)

    match parsed_args.subcommand:
        case None:
            parser.print_help()
        case "launch":
            __launch(
                project_name=parsed_args.project_name,
                port=parsed_args.port,
                open_browser=parsed_args.open_browser,
            )
        case "create":
            create_project(
                project_name=parsed_args.project_name,
                working_dir=parsed_args.working_dir,
            )
        case "quickstart":
            __quickstart()
        case _:
            # `parser.parse_args` raises an error if an unknown subcommand is passed,
            # so this case is impossible
            return
