"""Minimal local dev server for the image analyzer web UI.

Run from the repo root:
    uv run python src/server.py

Then open http://localhost:8080
"""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()
_analyzer = None
_analyzer_lock = __import__("threading").Lock()


def _warm_up():
    """Import heavy modules and init the analyzer in a background thread."""
    _get_analyzer()


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                from product_categorizer.image_analyzer import ImageAnalyzer
                _analyzer = ImageAnalyzer()
    return _analyzer


@app.on_event("startup")
async def startup():
    __import__("threading").Thread(target=_warm_up, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


def _analyze_one(image_url: str) -> float | None:
    """Analyze a single image and return its creative score, or None on failure."""
    try:
        result = _get_analyzer().analyze_image(image_url)
        score = result.get("img_overall_creative_score")
        return float(score) if score is not None else None
    except Exception:
        logging.getLogger(__name__).exception("Failed to analyze %s", image_url)
        return None


@app.post("/invoke")
async def invoke(request: Request):
    body = await request.json()
    params = body.get("parameters", body)
    image_url = params.get("image_url", "")
    if not image_url:
        return JSONResponse({"error": "image_url is required"}, status_code=400)
    result = _get_analyzer().analyze_image(image_url)
    return JSONResponse({"success": True, "outputs": {"output": result}})


@app.post("/bq")
async def bq_remote_function(request: Request):
    """BigQuery Remote Function endpoint (also available at POST /)."""
    return await _handle_bq(request)


@app.post("/")
async def root_post(request: Request):
    """Root POST handler for BigQuery Remote Function calls.

    BQ remote functions only support the root URL (no path).
    Detects BQ payload by the presence of "calls" key.
    """
    return await _handle_bq(request)


async def _handle_bq(request: Request):
    """Process a BigQuery Remote Function request.

    Receives: {"calls": [["url1"], ["url2"], ...]}
    Returns:  {"replies": [score1, score2, ...]}
    """
    body = await request.json()
    calls = body.get("calls", [])
    urls = [call[0] if call else "" for call in calls]

    replies: list[float | None] = [None] * len(urls)

    with ThreadPoolExecutor(max_workers=10) as pool:
        future_to_idx = {
            pool.submit(_analyze_one, url): i
            for i, url in enumerate(urls)
            if url
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            replies[idx] = future.result()

    return JSONResponse({"replies": replies})


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
