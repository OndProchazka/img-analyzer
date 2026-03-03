"""
ZenML deployment pipeline: Product Image Analysis HTTP service + UI.

Deploy locally (run from src/):
    cd src
    zenml stack set local-compute-remote-storage
    uv run zenml pipeline deploy img_analysis_demo_pipeline.img_analysis_demo_pipeline \
        --name product-categorizer-demo

Invoke:
    curl -X POST http://localhost:8080/invoke \
      -H "Content-Type: application/json" \
      -d '{"parameters": {"image_url": "https://example.com/img.jpg", "mode": "all"}}'

Deploy to Cloud Run (requires deployer-cloud-run in active stack):
    zenml stack set dev-stack
    uv run zenml pipeline deploy src.img_analysis_demo_pipeline.img_analysis_demo_pipeline \
        --name product-categorizer-demo
"""

import base64
import os

from zenml import pipeline, step
from zenml.config import DeploymentSettings, DockerSettings
from zenml.config.deployment_settings import MiddlewareSpec, SecureHeadersConfig
from zenml.steps import get_step_context

_deployment_password = os.environ.get("DEPLOYMENT_PASSWORD", "")


class BasicAuthMiddleware:
    """ASGI middleware: requires a password (any username) via HTTP Basic auth.

    Set DEMO_PASSWORD env var before deploying (default: "demo").
    """

    def __init__(self, app) -> None:
        self.app = app
        password = os.environ.get("DEPLOYMENT_PASSWORD")
        if not password:
            raise RuntimeError("DEPLOYMENT_PASSWORD env var is not set")
        self.password = password

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # Let health/metrics pass through unauthenticated
        if scope.get("path") in ("/health", "/metrics", "/info"):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode()
        if auth.lower().startswith("basic "):
            try:
                _, pwd = base64.b64decode(auth[6:]).decode().split(":", 1)
                if pwd == self.password:
                    await self.app(scope, receive, send)
                    return
            except Exception:
                pass

        async def _send_401(receive, send):
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"www-authenticate", b'Basic realm="Product Categorizer"'),
                    (b"content-type", b"text/plain"),
                    (b"content-length", b"12"),
                ],
            })
            await send({"type": "http.response.body", "body": b"Unauthorized"})

        await _send_401(receive, send)


def _init_analyzer():
    """Pre-load ImageAnalyzer once per deployment (taxonomy JSON + Gemini client)."""
    import sys
    from pathlib import Path

    # Ensure src/ is on the path when the daemon loads this hook
    src_dir = str(Path(__file__).parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from product_categorizer.image_analyzer import ImageAnalyzer

    return ImageAnalyzer()


@step
def categorize(image_url: str, mode: str = "all") -> dict:
    """Analyze a single product image URL and return the enriched result dict."""
    if not image_url:
        return {"analysis_error": "image_url is required"}
    # pipeline_state is set by on_init during deployment; fall back for local runs
    analyzer = get_step_context().pipeline_state or _init_analyzer()
    return analyzer.analyze_image(image_url, taxonomy_mode=mode)


@pipeline(
    on_init=_init_analyzer,
    settings={
        "docker": DockerSettings(
            warn_about_plain_text_secrets=False,
            environment={"DEPLOYMENT_PASSWORD": _deployment_password},
            pyproject_path="../pyproject.toml",
        ),
        "deployment": DeploymentSettings(
            app_title="Product Image Categorizer",
            dashboard_files_path="web",
            # Allow external product images in the preview (override img-src)
            secure_headers=SecureHeadersConfig(
                csp=(
                    "default-src 'none'; "
                    "script-src 'self' 'unsafe-inline'; "
                    "connect-src 'self'; "
                    "img-src * data:; "
                    "style-src 'self' 'unsafe-inline'; "
                    "base-uri 'self'; "
                    "form-action 'self'; "
                    "font-src 'self'; "
                    "frame-src 'self';"
                )
            ),
            custom_middlewares=[MiddlewareSpec(middleware=BasicAuthMiddleware)],
            uvicorn_host="0.0.0.0",
            uvicorn_port=8080,
        )
    },
)
def img_analysis_demo_pipeline(
    image_url: str = "http://img0.fbcdn.cz/tedi/api/stored/12/e5/12e5e78ed6081c2997564f00b8372066.png",
    mode: str = "two_step",
) -> dict:
    """HTTP-deployable pipeline: accepts image_url + mode, returns analysis dict."""
    return categorize(image_url=image_url, mode=mode)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_url", nargs="?", default="http://img0.fbcdn.cz/tedi/api/stored/12/e5/12e5e78ed6081c2997564f00b8372066.png")
    parser.add_argument("--mode", default="two_step")
    args = parser.parse_args()

    img_analysis_demo_pipeline(image_url=args.image_url, mode=args.mode)
