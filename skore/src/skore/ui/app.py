"""FastAPI factory used to create the API to interact with stores."""

import sys
from typing import Optional

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import Lifespan

from skore.project import Project, open
from skore.ui.dependencies import get_static_path
from skore.ui.project_routes import router as project_router


def create_app(
    project: Optional[Project] = None, lifespan: Optional[Lifespan] = None
) -> FastAPI:
    """FastAPI factory used to create the API to interact with `stores`."""
    app = FastAPI(lifespan=lifespan)

    # Give the app access to the project
    if not project:
        project = open("project.skore")

    app.state.project = project

    # Enable CORS support on all routes, for all origins and methods.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers from bottom to top.
    # Include routers always after all routes have been defined/imported.
    router = APIRouter(prefix="/api")
    router.include_router(project_router)

    # Include all sub routers.
    app.include_router(router)

    # Mount skore-ui from the static directory.
    # Should be after the API routes to avoid shadowing previous routes.
    static_path = get_static_path()
    if static_path.exists():
        # The mimetypes module may fail to set the
        # correct MIME type for javascript files.
        # More info on this here: https://github.com/encode/starlette/issues/829
        # So force it...
        if "win" in sys.platform:
            import mimetypes

            mimetypes.add_type("application/javascript", ".js")

        app.mount(
            "/",
            StaticFiles(
                directory=static_path,
                html=True,
                follow_symlink=True,
            ),
            name="static",
        )
    else:

        async def read_index(request):
            return HTMLResponse(
                """
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>skore-UI not found</title>
  <style>
    body {
      height: 100dvh;
      width: 100dvw;
      display: flex;
      justify-content: center;
      align-items: center;
      margin: 0;
      font-family: -apple-system,
        BlinkMacSystemFont,
        "Segoe UI", Roboto,
        Helvetica, Arial, sans-serif;
      background-color: #f5f5f5;
      color: #333;
    }

    .not-found {
      width: 60dvw;
      text-wrap: balance;
      background: white;
      padding: 2rem;
      border-radius: 8px;
      box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
      animation: fadeIn 0.5s ease-out;
      text-align: center;
    }


    .command {
      position: relative;

      pre {
        background: #f0f0f0;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        overflow-x: auto;
      }

      code {
        font-family: "SF Mono", Consolas, Monaco, monospace;
        color: #d63384;
      }

      .copy-button {
        position: absolute;
        right: 5px;
        top: 50%;
        transform: translateY(-50%);
        border: none;
        transition: background-color linear 0.3s;
        background-color: #f0f0f0;
        padding: 0.1rem;

        &:hover {
          background-color: #e6e6e6;
        }
      }
    }

    @keyframes fadeIn {
      from {
        opacity: 0;
        transform: translateY(-20px);
      }

      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    @media (max-width: 768px) {
      .not-found {
        width: 80dvw;
      }
    }
  </style>
</head>

<body>
  <div class="not-found">
    <h1 style="margin-top: 0;">skore-UI Missing</h1>
    <p>skore-UI has not been built locally.</p>
    <p>To build it, you'll need to install
      <a href="https://nodejs.org/en/download">node</a>,
      then to run:
    </p>
    <div class="command">
      <pre><code>make build-skore-ui</code></pre>
      <button class="copy-button" onclick="copyCode()" title="copy">📑</button>
    </div>
    <p>and restart the server.</p>
  </div>

  <script>
    function copyCode() {
      const code = document.querySelector('code').textContent;
      navigator.clipboard.writeText(code);
    }
  </script>
</body>

</html>
""",
                status_code=404,
            )

        app.router.add_route("/", read_index, methods=["GET"])

    return app
