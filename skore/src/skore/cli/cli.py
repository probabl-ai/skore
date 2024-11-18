"""CLI for Skore."""

import argparse
import pathlib
import sys
from importlib.metadata import version
from typing import Optional

from skore.cli.launch_dashboard import __launch
from skore.cli.quickstart_command import __quickstart
from skore.project import create


def cli(args: Optional[list[str]] = None):
    """CLI for Skore."""
    parser = argparse.ArgumentParser(prog="skore")

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

    parser_create = subparsers.add_parser("create", help="Create a project")
    parser_create.add_argument(
        "project_name",
        nargs="?",
        help="the name or path of the project to create (default: %(default)s)",
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
    parser_create.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite an existing project with the same name",
    )

    subparsers.add_parser(
        "quickstart", help='Create a "project.skore" file and start the UI'
    )

    if args is None:
        args = sys.argv[1:]
    parsed_args: argparse.Namespace = parser.parse_args(args)

    try:
        if parsed_args.subcommand == "launch":
            __launch(
                project_name=parsed_args.project_name,
                port=parsed_args.port,
                open_browser=parsed_args.open_browser,
            )
        elif parsed_args.subcommand == "create":
            create(
                project_name=parsed_args.project_name,
                working_dir=parsed_args.working_dir,
                overwrite=parsed_args.overwrite,
            )
        elif parsed_args.subcommand == "quickstart":
            __quickstart()
        else:
            parser.print_help()
        return 0
    except Exception:
        raise
        return 1
