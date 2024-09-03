"""CLI for Mandr."""

import argparse

from mandr.dashboard.dashboard import Dashboard


def cli(args: list[str]):
    """CLI for Mandr."""
    parser = argparse.ArgumentParser(
        prog="mandr",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    parser_dashboard = subparsers.add_parser("dashboard", help="Start the dashboard")
    parser_dashboard.add_argument(
        "--port",
        type=int,
        help="the port at which to bind the UI server (default: %(default)s)",
        default=22140,
    )
    parser_dashboard.add_argument(
        "--open-browser",
        action=argparse.BooleanOptionalAction,
        help=(
            "whether to automatically open a browser tab showing the dashboard "
            "(default: %(default)s)"
        ),
        default=True,
    )

    parsed_args: argparse.Namespace = parser.parse_args(args)

    match parsed_args.subcommand:
        case None:
            parser.print_help()
        case "dashboard":
            Dashboard(port=parsed_args.port).open(open_browser=parsed_args.open_browser)
        case _:
            # `parser.parse_args` raises an error if an unknown subcommand is passed,
            # so this case is impossible
            return
