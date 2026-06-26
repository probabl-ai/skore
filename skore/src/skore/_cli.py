import argparse

from skore._plugins.local.project import init_workspace


def init_workspace_command(args):
    workspace = init_workspace().resolve()
    print(f"Skore workspace created: {workspace}")


def main():
    parser = argparse.ArgumentParser(
        description="skore local init: create a workspace in the current directory",
    )
    subparsers = parser.add_subparsers(title="commands", required=True)
    local_parser = subparsers.add_parser(
        "local",
        help="commands for managing local workspaces",
        description="commands for managing local workspaces",
    )
    local_subparsers = local_parser.add_subparsers(
        title="sub-commands", help="local workspace commands", required=True
    )
    local_init_parser = local_subparsers.add_parser(
        "init",
        help="create a workspace in the current directory",
        description="create a workspace in the current directory",
    )
    local_init_parser.set_defaults(func=init_workspace_command)
    args = parser.parse_args()
    args.func(args)
