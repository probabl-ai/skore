"""Server module to run the FastAPI application."""

import argparse

import uvicorn

from skore.project.project import Project
from skore.ui.app import create_app


def run_server(port: int, project_path: str) -> uvicorn.Server:
    """Run the uvicorn server with the given project and port.

    Parameters
    ----------
    port : int
        The port number to run the server on.
    project_path : str
        Path to the skore project to load.

    Returns
    -------
    uvicorn.Server
        The uvicorn server instance.
    """
    project = Project(project_path, if_exists="load")
    app = create_app(project=project)

    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    server.run()
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the skore UI server")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--project-path", type=str, required=True)
    args = parser.parse_args()
    run_server(args.port, args.project_path)
