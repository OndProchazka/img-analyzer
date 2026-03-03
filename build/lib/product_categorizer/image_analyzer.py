import io
import json
import logging
import os
import random
import threading
import time
from collections import Counter
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

GCP_PROJECT = "martin-panacek-playground"
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
    language_clarity_issues: list[str]
    product_name_detected: str | None = None
    product_name_consistency_ok: bool
    product_type: str  # concise description of what the product IS, e.g. "running shoes", "laptop stand"
    seasonality: str | None = (
        None  # "Spring/Summer" | "Autumn/Winter" | "All-season" | null if not applicable (e.g. electronics)
    )
    target_gender: str  # "men" | "women" | "kids" | "neutral" (neutral = not gender-relevant)
    overall_notes: str


TAXONOMY_PATH = Path(__file__).parent / "google_taxonomy.json"


class _TaxonomyPick(BaseModel):
    category_id: str
    category_path: str


class _L1Pick(BaseModel):
    category: str


class TaxonomyClassifier:
    def __init__(self, taxonomy_path: Path = TAXONOMY_PATH):
        id_to_path: dict[str, str] = json.loads(taxonomy_path.read_text())
        paths = set(id_to_path.values())
        # Keep only leaf nodes (no other entry starts with this path + " >")
        self._leaves: dict[str, str] = {
            k: v for k, v in id_to_path.items() if not any(p.startswith(v + " >") for p in paths)
        }
        self._leaves_text = "\n".join(f"{k}: {v}" for k, v in sorted(self._leaves.items(), key=lambda x: x[1]))
        # Precompute L1 groups for two-step approach
        self._l1_categories: list[str] = sorted(set(v.split(" > ")[0] for v in self._leaves.values()))
        self._leaves_by_l1: dict[str, dict[str, str]] = {
            l1: {k: v for k, v in self._leaves.items() if v == l1 or v.startswith(l1 + " >")}
            for l1 in self._l1_categories
        }

    # ------------------------------------------------------------------ #
    # Approach 1: leaf-first — all 4719 leaves in one shot               #
    # ------------------------------------------------------------------ #
    def classify_leaf_first(
        self,
        image_bytes: bytes,
        mime_type: str,
        product_description: str,
        client: genai.Client,
    ) -> tuple[str, str]:
        r = _gemini_generate(
            client,
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                _load_prompt("taxonomy_leaf_first.md").format(
                    product_description=product_description,
                    leaves_text=self._leaves_text,
                ),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_TaxonomyPick,
                temperature=0.0,
            ),
        )
        return self._resolve(_TaxonomyPick.model_validate_json(r.text), self._leaves)

    # ------------------------------------------------------------------ #
    # Approach 2: two-step — pick L1 first, then leaf within branch      #
    # ------------------------------------------------------------------ #
    def classify_two_step(
        self,
        image_bytes: bytes,
        mime_type: str,
        product_description: str,
        client: genai.Client,
    ) -> tuple[str, str]:
        l1_list = "\n".join(self._l1_categories)
        r1 = _gemini_generate(
            client,
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                _load_prompt("taxonomy_two_step_l1.md").format(
                    product_description=product_description,
                    l1_list=l1_list,
                ),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_L1Pick,
                temperature=0.0,
            ),
        )
        l1 = _L1Pick.model_validate_json(r1.text).category
        branch = self._leaves_by_l1.get(l1) or next(
            (v for k, v in self._leaves_by_l1.items() if k.startswith(l1) or l1.startswith(k)),
            None,
        )
        if not branch:
            return self.classify_leaf_first(image_bytes, mime_type, product_description, client)

        branch_text = "\n".join(f"{k}: {v}" for k, v in sorted(branch.items(), key=lambda x: x[1]))
        r2 = _gemini_generate(
            client,
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                _load_prompt("taxonomy_two_step_leaf.md").format(
                    product_description=product_description,
                    branch_text=branch_text,
                ),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_TaxonomyPick,
                temperature=0.0,
            ),
        )
        return self._resolve(_TaxonomyPick.model_validate_json(r2.text), branch)

    # ------------------------------------------------------------------ #
    # Approach 3: keyword filter — narrow by product_type, then pick     #
    # ------------------------------------------------------------------ #
    def classify_keyword(
        self,
        image_bytes: bytes,
        mime_type: str,
        product_type: str,
        client: genai.Client,
    ) -> tuple[str, str]:
        keywords = [w for w in product_type.lower().split() if len(w) > 2]
        candidates = {k: v for k, v in self._leaves.items() if any(kw in v.lower() for kw in keywords)}
        if not candidates:
            return self.classify_leaf_first(image_bytes, mime_type, product_type, client)
        if len(candidates) == 1:
            k, v = next(iter(candidates.items()))
            return k, v
        candidates_text = "\n".join(f"{k}: {v}" for k, v in sorted(candidates.items(), key=lambda x: x[1]))
        r = _gemini_generate(
            client,
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                _load_prompt("taxonomy_keyword.md").format(
                    product_type=product_type,
                    candidates_text=candidates_text,
                ),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_TaxonomyPick,
                temperature=0.0,
            ),
        )
        return self._resolve(_TaxonomyPick.model_validate_json(r.text), candidates)

    # ------------------------------------------------------------------ #
    # Run all three in parallel                                           #
    # ------------------------------------------------------------------ #
    def classify_all(
        self,
        image_bytes: bytes,
        mime_type: str,
        product_description: str,
        product_type: str,
        client: genai.Client,
    ) -> dict[str, tuple[str | None, str | None]]:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                "leaf_first": executor.submit(
                    self.classify_leaf_first, image_bytes, mime_type, product_description, client
                ),
                "two_step": executor.submit(
                    self.classify_two_step, image_bytes, mime_type, product_description, client
                ),
                "keyword": executor.submit(self.classify_keyword, image_bytes, mime_type, product_type, client),
            }
        results: dict[str, tuple[str | None, str | None]] = {}
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except Exception as e:
                logger.warning("Taxonomy %s failed: %s", name, e)
                results[name] = (None, None)
        return results

    def _resolve(self, pick: _TaxonomyPick, candidates: dict[str, str]) -> tuple[str, str]:
        if pick.category_id in candidates:
            return pick.category_id, candidates[pick.category_id]
        reverse = {v: k for k, v in candidates.items()}
        if pick.category_path in reverse:
            return reverse[pick.category_path], pick.category_path
        return pick.category_id, pick.category_path


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
        self._taxonomy = TaxonomyClassifier()

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
        result["img_language_clarity_issues"] = "; ".join(analysis.language_clarity_issues)
        result["img_product_name"] = analysis.product_name_detected
        result["img_product_name_ok"] = analysis.product_name_consistency_ok
        result["img_product_type"] = analysis.product_type
        result["img_seasonality"] = analysis.seasonality
        result["img_target_gender"] = analysis.target_gender
        result["img_overall_notes"] = analysis.overall_notes

        try:
            product_description = " ".join(filter(None, [analysis.product_type, analysis.product_name_detected]))

            if taxonomy_mode == "all":
                taxonomy_results = self._taxonomy.classify_all(
                    image_bytes, mime_type, product_description, analysis.product_type, self._client
                )
            else:
                _single: dict[str, tuple[str | None, str | None]] = {
                    "leaf_first": lambda: self._taxonomy.classify_leaf_first(  # type: ignore[misc]
                        image_bytes, mime_type, product_description, self._client
                    ),
                    "two_step": lambda: self._taxonomy.classify_two_step(  # type: ignore[misc]
                        image_bytes, mime_type, product_description, self._client
                    ),
                    "keyword": lambda: self._taxonomy.classify_keyword(  # type: ignore[misc]
                        image_bytes, mime_type, analysis.product_type, self._client
                    ),
                }
                if taxonomy_mode not in _single:
                    raise ValueError(f"Unknown taxonomy_mode: {taxonomy_mode!r}")
                taxonomy_results = {k: (None, None) for k in ("leaf_first", "two_step", "keyword")}
                taxonomy_results[taxonomy_mode] = _single[taxonomy_mode]()

            for approach, (tid, tpath) in taxonomy_results.items():
                result[f"img_cat_{approach}_id"] = tid
                result[f"img_cat_{approach}_path"] = tpath

            # Consensus: most common non-None path across approaches
            paths = [p for _, p in taxonomy_results.values() if p]
            if paths:
                consensus_path = Counter(paths).most_common(1)[0][0]
                reverse = {v: k for k, v in self._taxonomy._leaves.items()}
                result["img_category_id"] = reverse.get(
                    consensus_path, taxonomy_results.get("two_step", (None, None))[0]
                )
                result["img_category_path"] = consensus_path
            else:
                result["img_category_id"] = None
                result["img_category_path"] = None
        except Exception as e:
            logger.warning("Taxonomy classification failed for %s: %s", url, e)
            for approach in ("leaf_first", "two_step", "keyword"):
                result[f"img_cat_{approach}_id"] = None
                result[f"img_cat_{approach}_path"] = None
            result["img_category_id"] = None
            result["img_category_path"] = None

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
