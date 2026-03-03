import io
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import google.auth
import google.auth.credentials
import pandas as pd
import requests
from google import genai
from google.cloud import vision as gvision
from google.genai import types
from google.oauth2 import service_account
from PIL import Image as PILImage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

GCP_PROJECT = "prochazka-ml-playground"
GEMINI_MODEL = "gemini-3-flash-preview"
MAX_WORKERS = 5
REQUEST_TIMEOUT = 30

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text()


# Limit concurrent Gemini calls to avoid rate limits on the preview model
_GEMINI_SEM = threading.Semaphore(2)


def _gemini_generate(client: "genai.Client", **kwargs):
    """Call client.models.generate_content with a semaphore and exponential backoff on 429."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with _GEMINI_SEM:
                return client.models.generate_content(**kwargs)
        except Exception as e:
            if "429" not in str(e) and "RESOURCE_EXHAUSTED" not in str(e):
                raise
            if attempt == max_retries - 1:
                raise
            wait = min(2.0 * (1.5**attempt) + random.uniform(0, 1), 30.0)
            logger.warning("Gemini 429 (attempt %d/%d), retrying in %.1fs", attempt + 1, max_retries, wait)
            time.sleep(wait)


SYSTEM_PROMPT = _load_prompt("analysis_system.md")
USER_PROMPT_TEMPLATE = _load_prompt("analysis_user.md")


class ImageAnalysis(BaseModel):
    # Visual quality
    visual_hierarchy_first_element: str  # product | discount | logo | text | other
    text_readability_mobile_score: float
    text_contrast_vs_background_score: float
    text_contrast_vs_product_score: float
    whitespace_score: float
    attention_focal_point_score: float
    attention_hierarchy_score: float
    attention_contrast_score: float
    background_blend_risk: str  # low | medium | high
    background_blend_explanation: str | None = None
    crop_recommendation: str | None = None
    # Anchoring
    has_price: bool
    detected_price: str | None = None
    has_discount_pct: bool
    detected_discount_pct: str | None = None  # e.g. "-20%"
    has_before_after_price: bool
    has_coupon: bool
    detected_coupon: str | None = None
    language_clarity_score: float
    message_headline_clarity_score: float
    message_cta_clarity_score: float
    message_text_density_score: float
    language_clarity_issues: list[str]
    product_name_detected: str | None = None
    product_name_consistency_ok: bool
    product_type: str  # concise description of what the product IS, e.g. "running shoes", "laptop stand"
    seasonality: str | None = (
        None  # "Spring/Summer" | "Autumn/Winter" | "All-season" | null if not applicable (e.g. electronics)
    )
    target_gender: str  # "men" | "women" | "kids" | "neutral" (neutral = not gender-relevant)
    overall_notes: str
    # Offer & branding
    offer_prominence_score: float
    offer_relevance_score: float
    branding_logo_visibility_score: float
    branding_distinctiveness_score: float
    # Product presentation & multi-view
    product_clarity_score: float
    product_context_fit_score: float
    multi_view_has_multiple_shots: bool
    multi_view_num_shots: int
    multi_view_layout_type: str
    product_multi_view_complementarity_score: float
    product_multi_view_clarity_score: float
    product_multi_view_layout_efficiency_score: float
    # Simplicity, emotion, craft
    clutter_score: float
    emotional_resonance_score: float
    aesthetic_craft_score: float


def _fetch_image(url: str) -> bytes:
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.content


def _get_mime_type(image_bytes: bytes) -> str:
    img = PILImage.open(io.BytesIO(image_bytes))
    return {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "GIF": "image/gif",
        "WEBP": "image/webp",
    }.get(img.format, "image/jpeg")


def _load_credentials(
    key_file: str | Path | None,
    project_id: str,
) -> google.auth.credentials.Credentials:
    """Load credentials from a service account key file, or fall back to ADC."""
    if key_file:
        creds = service_account.Credentials.from_service_account_file(
            str(key_file),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    else:
        creds, _ = google.auth.default()
    return creds.with_quota_project(project_id)


def _vision_api_analyze(image_bytes: bytes, credentials: google.auth.credentials.Credentials) -> dict:
    client = gvision.ImageAnnotatorClient(credentials=credentials)
    image = gvision.Image(content=image_bytes)

    features = [
        gvision.Feature(type_=gvision.Feature.Type.TEXT_DETECTION),
        gvision.Feature(type_=gvision.Feature.Type.IMAGE_PROPERTIES),
        gvision.Feature(type_=gvision.Feature.Type.OBJECT_LOCALIZATION),
        gvision.Feature(type_=gvision.Feature.Type.LOGO_DETECTION),
    ]

    r = client.annotate_image(gvision.AnnotateImageRequest(image=image, features=features))

    result = {
        "ocr_text": "",
        "dominant_colors": [],
        "product_coverage_pct": None,
        "logo_detected": False,
        "logo_position": None,
    }

    if r.text_annotations:
        result["ocr_text"] = r.text_annotations[0].description.strip()

    if r.image_properties_annotation.dominant_colors.colors:
        colors = sorted(
            r.image_properties_annotation.dominant_colors.colors,
            key=lambda c: c.score,
            reverse=True,
        )
        result["dominant_colors"] = [
            f"rgb({int(c.color.red)},{int(c.color.green)},{int(c.color.blue)}) score={c.score:.2f}" for c in colors[:5]
        ]

    if r.localized_object_annotations:
        top = max(r.localized_object_annotations, key=lambda o: o.score)
        v = top.bounding_poly.normalized_vertices
        xs = [p.x for p in v]
        ys = [p.y for p in v]
        result["product_coverage_pct"] = round((max(xs) - min(xs)) * (max(ys) - min(ys)) * 100, 1)

    if r.logo_annotations:
        result["logo_detected"] = True
        logo = r.logo_annotations[0]
        img = PILImage.open(io.BytesIO(image_bytes))
        w, h = img.width, img.height
        verts = logo.bounding_poly.vertices
        if verts and w > 0 and h > 0:
            cx = sum(v.x for v in verts) / len(verts) / w
            cy = sum(v.y for v in verts) / len(verts) / h
            v_pos = "top" if cy < 0.4 else ("bottom" if cy > 0.6 else "middle")
            h_pos = "left" if cx < 0.4 else ("right" if cx > 0.6 else "center")
            result["logo_position"] = f"{v_pos}-{h_pos}"

    return result


def _gemini_analyze(
    image_bytes: bytes,
    mime_type: str,
    vision_data: dict,
    client: genai.Client,
) -> ImageAnalysis:
    prompt = USER_PROMPT_TEMPLATE.format(
        ocr_text=vision_data["ocr_text"] or "(no text detected)",
        dominant_colors=(
            ", ".join(vision_data["dominant_colors"]) if vision_data["dominant_colors"] else "(not available)"
        ),
    )

    response = _gemini_generate(
        client,
        model=GEMINI_MODEL,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ImageAnalysis,
            temperature=0.1,
        ),
    )

    return ImageAnalysis.model_validate_json(response.text)


class ImageAnalyzer:
    def __init__(
        self,
        project_id: str | None = None,
        region: str = "global",
        service_account_key: str | Path | None = None,
    ):
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", GCP_PROJECT)
        self._credentials = _load_credentials(service_account_key, self._project_id)
        self._client = genai.Client(
            vertexai=True,
            project=self._project_id,
            location=region,
            credentials=self._credentials,
        )

    def analyze_image(self, url: str, taxonomy_mode: str = "all") -> dict:
        result: dict = {"analysis_error": None}

        try:
            image_bytes = _fetch_image(url)
        except Exception as e:
            result["analysis_error"] = f"fetch_failed: {e}"
            return result

        mime_type = _get_mime_type(image_bytes)

        try:
            vision_data = _vision_api_analyze(image_bytes, self._credentials)
        except Exception as e:
            logger.warning("Vision API failed for %s: %s", url, e)
            vision_data = {
                "ocr_text": "",
                "dominant_colors": [],
                "product_coverage_pct": None,
                "logo_detected": False,
                "logo_position": None,
            }

        result["img_product_coverage_pct"] = vision_data["product_coverage_pct"]
        result["img_logo_detected"] = vision_data["logo_detected"]
        result["img_logo_position"] = vision_data["logo_position"]

        try:
            analysis = _gemini_analyze(image_bytes, mime_type, vision_data, self._client)
        except Exception as e:
            result["analysis_error"] = f"gemini_failed: {e}"
            return result

        result["img_visual_hierarchy"] = analysis.visual_hierarchy_first_element
        result["img_text_readability_mobile"] = analysis.text_readability_mobile_score
        result["img_text_contrast_bg"] = analysis.text_contrast_vs_background_score
        result["img_text_contrast_product"] = analysis.text_contrast_vs_product_score
        result["img_whitespace_score"] = analysis.whitespace_score
        result["img_attention_focal_point_score"] = analysis.attention_focal_point_score
        result["img_attention_hierarchy_score"] = analysis.attention_hierarchy_score
        result["img_attention_contrast_score"] = analysis.attention_contrast_score
        result["img_bg_blend_risk"] = analysis.background_blend_risk
        result["img_bg_blend_explanation"] = analysis.background_blend_explanation
        result["img_crop_recommendation"] = analysis.crop_recommendation
        result["img_has_price"] = analysis.has_price
        result["img_price_text"] = analysis.detected_price
        result["img_has_discount_pct"] = analysis.has_discount_pct
        result["img_discount_pct_text"] = analysis.detected_discount_pct
        result["img_has_before_after_price"] = analysis.has_before_after_price
        result["img_has_coupon"] = analysis.has_coupon
        result["img_coupon_text"] = analysis.detected_coupon
        result["img_language_clarity_score"] = analysis.language_clarity_score
        result["img_message_headline_clarity_score"] = analysis.message_headline_clarity_score
        result["img_message_cta_clarity_score"] = analysis.message_cta_clarity_score
        result["img_message_text_density_score"] = analysis.message_text_density_score
        result["img_language_clarity_issues"] = "; ".join(analysis.language_clarity_issues)
        result["img_product_name"] = analysis.product_name_detected
        result["img_product_name_ok"] = analysis.product_name_consistency_ok
        result["img_product_type"] = analysis.product_type
        result["img_seasonality"] = analysis.seasonality
        result["img_target_gender"] = analysis.target_gender
        result["img_overall_notes"] = analysis.overall_notes
        result["img_offer_prominence_score"] = analysis.offer_prominence_score
        result["img_offer_relevance_score"] = analysis.offer_relevance_score
        result["img_branding_logo_visibility_score"] = analysis.branding_logo_visibility_score
        result["img_branding_distinctiveness_score"] = analysis.branding_distinctiveness_score
        result["img_product_clarity_score"] = analysis.product_clarity_score
        result["img_product_context_fit_score"] = analysis.product_context_fit_score
        result["img_multi_view_has_multiple_shots"] = analysis.multi_view_has_multiple_shots
        result["img_multi_view_num_shots"] = analysis.multi_view_num_shots
        result["img_multi_view_layout_type"] = analysis.multi_view_layout_type
        result["img_product_multi_view_complementarity_score"] = (
            analysis.product_multi_view_complementarity_score
        )
        result["img_product_multi_view_clarity_score"] = analysis.product_multi_view_clarity_score
        result["img_product_multi_view_layout_efficiency_score"] = (
            analysis.product_multi_view_layout_efficiency_score
        )
        result["img_clutter_score"] = analysis.clutter_score
        result["img_emotional_resonance_score"] = analysis.emotional_resonance_score
        result["img_aesthetic_craft_score"] = analysis.aesthetic_craft_score

        # Composite scores (simple means of relevant sub-scores; weights can be tuned later)
        def _mean(values: list[float]) -> float:
            vals = [v for v in values if v is not None]
            return float(sum(vals) / len(vals)) if vals else 0.0

        result["img_attention_score"] = _mean(
            [
                analysis.attention_focal_point_score,
                analysis.attention_hierarchy_score,
                analysis.attention_contrast_score,
            ]
        )
        result["img_message_clarity_score_v2"] = _mean(
            [
                analysis.language_clarity_score,
                analysis.message_headline_clarity_score,
                analysis.message_cta_clarity_score,
                1.0 - analysis.message_text_density_score,
            ]
        )
        result["img_branding_score"] = _mean(
            [
                analysis.branding_logo_visibility_score,
                analysis.branding_distinctiveness_score,
            ]
        )
        result["img_offer_strength_score"] = _mean(
            [
                analysis.offer_prominence_score,
                analysis.offer_relevance_score,
            ]
        )
        result["img_product_presentation_score"] = _mean(
            [
                analysis.product_clarity_score,
                analysis.product_context_fit_score,
                analysis.product_multi_view_complementarity_score,
                analysis.product_multi_view_clarity_score,
                analysis.product_multi_view_layout_efficiency_score,
            ]
        )
        result["img_overall_creative_score"] = _mean(
            [
                result["img_attention_score"],
                result["img_message_clarity_score_v2"],
                result["img_branding_score"],
                result["img_offer_strength_score"],
                result["img_product_presentation_score"],
                analysis.emotional_resonance_score,
                analysis.aesthetic_craft_score,
                1.0 - analysis.clutter_score,
            ]
        )

        return result

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        image_url_col: str = "image_url",
        max_workers: int = MAX_WORKERS,
    ) -> pd.DataFrame:
        urls = df[image_url_col].tolist()
        results: list = [None] * len(urls)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.analyze_image, url): i for i, url in enumerate(urls)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = {"analysis_error": str(e)}

        analysis_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), analysis_df], axis=1)
