import argparse

import pandas as pd

from product_categorizer.image_analyzer import ImageAnalyzer


def analyze_product_images(
    df: pd.DataFrame,
    image_url_col: str = "image_url",
    project_id: str | None = None,
    region: str = "global",
    max_workers: int = 5,
    service_account_key: str | None = None,
) -> pd.DataFrame:
    """
    Analyze product images in a DataFrame and return the enriched DataFrame.

    Uses GOOGLE_CLOUD_PROJECT env var for project_id if not provided.
    Authentication via GCP Application Default Credentials.

    Args:
        df: Input DataFrame with at least one image URL column.
        image_url_col: Name of the column containing image URLs.
        project_id: GCP project ID (falls back to GOOGLE_CLOUD_PROJECT env var).
        region: Vertex AI region where Claude is deployed.
        max_workers: Number of concurrent image analyses.

    Returns:
        Original DataFrame with img_* analysis columns appended.
    """
    analyzer = ImageAnalyzer(project_id=project_id, region=region, service_account_key=service_account_key)
    return analyzer.analyze_dataframe(df, image_url_col=image_url_col, max_workers=max_workers)


def _val(x):
    """Return None if x is NaN/None, otherwise return x."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return x


def _print_results(result: pd.DataFrame) -> None:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    for i, row in result.iterrows():
        url = row.get("image_url", f"image {i}")
        error = _val(row.get("analysis_error"))

        console.print(f"\n[bold cyan]Image:[/] {url}")

        if error:
            console.print(f"[bold red]Error:[/] {error}")
            continue

        # Visual
        visual = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
        visual.add_column("field", style="dim", width=24)
        visual.add_column("value")
        visual.add_row("hierarchy", str(row["img_visual_hierarchy"]))
        visual.add_row("readability mobile", f"{row['img_text_readability_mobile']:.2f}")
        visual.add_row("contrast vs bg", f"{row['img_text_contrast_bg']:.2f}")
        visual.add_row("contrast vs product", f"{row['img_text_contrast_product']:.2f}")
        visual.add_row("whitespace", f"{row['img_whitespace_score']:.2f}")
        visual.add_row(
            "product coverage", f"{row['img_product_coverage_pct']}%" if row["img_product_coverage_pct"] else "—"
        )
        blend = row["img_bg_blend_risk"]
        blend_color = {"low": "green", "medium": "yellow", "high": "red"}.get(blend, "white")
        visual.add_row("bg blend risk", f"[{blend_color}]{blend}[/]")
        if _val(row.get("img_bg_blend_explanation")):
            visual.add_row("blend note", row["img_bg_blend_explanation"])
        if _val(row.get("img_crop_recommendation")):
            visual.add_row("crop reco", row["img_crop_recommendation"])
        console.print(Panel(visual, title="[bold]Visual", border_style="blue"))

        # Anchoring
        anch = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
        anch.add_column("field", style="dim", width=24)
        anch.add_column("value")

        def yn(val, detail=None):
            detail = _val(detail)
            # Use ASCII-only markers to avoid Unicode issues on some Windows consoles
            mark = "[green]YES[/]" if val else "[red]NO[/]"
            return f"{mark}  {detail}" if (val and detail) else mark

        anch.add_row("price", yn(row["img_has_price"], row.get("img_price_text")))
        anch.add_row("discount %", yn(row["img_has_discount_pct"], row.get("img_discount_pct_text")))
        anch.add_row("before/after price", yn(row["img_has_before_after_price"]))
        anch.add_row("coupon", yn(row["img_has_coupon"], row.get("img_coupon_text")))
        anch.add_row("language clarity", f"{row['img_language_clarity_score']:.2f}")
        if _val(row.get("img_language_clarity_issues")):
            anch.add_row("language issues", row["img_language_clarity_issues"])
        anch.add_row("product type", str(_val(row.get("img_product_type")) or "—"))
        if _val(row.get("img_seasonality")):
            anch.add_row("seasonality", str(row["img_seasonality"]))
        name_status = "[green]ok[/]" if row.get("img_product_name_ok") else "[red]inconsistent[/]"
        anch.add_row("product name", f"{_val(row.get('img_product_name')) or '—'}  {name_status}")
        console.print(Panel(anch, title="[bold]Anchoring", border_style="blue"))

        # Notes
        if _val(row.get("img_overall_notes")):
            console.print(Panel(row["img_overall_notes"], title="[bold]Notes", border_style="dim"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze product image(s)")
    parser.add_argument("urls", nargs="+", help="Image URL(s) to analyze")
    parser.add_argument("--project", default=None, help="GCP project ID")
    parser.add_argument("--region", default="global")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--sa-key", default=None, help="Path to service account JSON key file")
    args = parser.parse_args()

    df = pd.DataFrame({"image_url": args.urls})
    result = analyze_product_images(
        df,
        project_id=args.project,
        region=args.region,
        max_workers=args.workers,
        service_account_key=args.sa_key,
    )
    _print_results(result)
