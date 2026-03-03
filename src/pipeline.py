"""
ZenML pipeline for batch product image analysis.

The active ZenML stack controls where the pipeline runs:
    zenml stack set default        # local
    zenml stack set dev-stack      # Vertex AI

Run:
    uv run python src/pipeline.py
"""

import argparse

import pandas as pd
from zenml import pipeline, step
from zenml.logger import get_logger

from product_categorizer.image_analyzer import ImageAnalyzer

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Steps                                                                        #
# --------------------------------------------------------------------------- #


@step(runtime="inline")
def load_catalog(image_urls: list[str]) -> pd.DataFrame:
    """Build the input DataFrame from a list of image URLs."""
    logger.info("Loading %d image URLs", len(image_urls))
    return pd.DataFrame({"image_url": image_urls})


@step(runtime="inline")
def analyze_images(
    df: pd.DataFrame,
    region: str = "global",
    max_workers: int = 5,
) -> pd.DataFrame:
    """Run Vision API + Gemini analysis on every image in the DataFrame."""
    logger.info("Analyzing %d images (max_workers=%d)", len(df), max_workers)
    analyzer = ImageAnalyzer(region=region)
    result = analyzer.analyze_dataframe(df, max_workers=max_workers)
    errors = result["analysis_error"].notna().sum()
    logger.info("Analysis complete — %d succeeded, %d errors", len(result) - errors, errors)
    return result


@step(runtime="inline")
def quality_report(df: pd.DataFrame) -> dict:
    """
    Summarize image quality metrics across the catalog.
    Returns a dict that ZenML stores as an artifact.
    """
    ok = df[df["analysis_error"].isna()]
    total = len(df)
    analyzed = len(ok)

    report = {
        "total_images": total,
        "analyzed": analyzed,
        "errors": total - analyzed,
        # Blend risk breakdown
        "blend_risk_high": int((ok["img_bg_blend_risk"] == "high").sum()),
        "blend_risk_medium": int((ok["img_bg_blend_risk"] == "medium").sum()),
        "blend_risk_low": int((ok["img_bg_blend_risk"] == "low").sum()),
        # Average scores
        "avg_readability_mobile": round(ok["img_text_readability_mobile"].mean(), 3),
        "avg_text_contrast_bg": round(ok["img_text_contrast_bg"].mean(), 3),
        "avg_whitespace": round(ok["img_whitespace_score"].mean(), 3),
        # Anchoring coverage
        "pct_has_price": round(ok["img_has_price"].mean() * 100, 1),
        "pct_has_discount": round(ok["img_has_discount_pct"].mean() * 100, 1),
        # Seasonality breakdown
        "seasonality_counts": ok["img_seasonality"].value_counts().to_dict(),
    }

    logger.info("Quality report:\n%s", "\n".join(f"  {k}: {v}" for k, v in report.items()))
    return report


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #


@pipeline(dynamic=True)
def product_image_pipeline(
    image_urls: list[str],
    region: str = "global",
    max_workers: int = 5,
):
    df = load_catalog(image_urls=image_urls)
    analyzed = analyze_images(df=df, region=region, max_workers=max_workers)
    quality_report(df=analyzed)


# --------------------------------------------------------------------------- #
# Entrypoint                                                                   #
# --------------------------------------------------------------------------- #

SAMPLE_URLS = [
    "https://image.alza.cz/products/RI053b2/RI053b2.jpg?width=1000&height=1000",
    "https://image.alza.cz/products/JA940k4a/JA940k4a.jpg?width=1000&height=1000",
    "https://image.alza.cz/products/JA040se25n3/JA040se25n3.jpg?width=1000&height=1000",
    "https://image.alza.cz/products/SGR_BF_A251S/SGR_BF_A251S.jpg?width=1000&height=1000",
    "https://image.alza.cz/products/AC_SB_25B15/AC_SB_25B15.jpg?width=1000&height=1000",
    "https://assets.burberry.com/is/image/Burberryltd/3EC90CBA-9383-45BF-B671-DE4DC12B5A68?$BBY_V3_SL_1$&wid=2500&hei=2500",
    "https://assets.burberry.com/is/image/Burberryltd/485EE0FE-6C4D-40BA-B2D4-064103D05735?$BBY_V3_SL_1$&wid=2500&hei=2500",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the product image analysis pipeline")
    parser.add_argument("--region", default="global")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    from zenml.config import DockerSettings

    product_image_pipeline.with_options(
        settings={
            "docker": DockerSettings(
                python_package_installer="uv",
                pyproject_path="../pyproject.toml",
            ),
            "orchestrator": {"synchronous": False},
        },
        enable_cache=False,
    )(
        image_urls=SAMPLE_URLS,
        region=args.region,
        max_workers=args.workers,
    )
