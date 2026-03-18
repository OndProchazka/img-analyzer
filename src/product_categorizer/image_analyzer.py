from __future__ import annotations

import io
import logging
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import google.auth
import google.auth.credentials
import requests
from google import genai
from google.cloud import vision as gvision
from google.genai import types
from google.oauth2 import service_account
from PIL import Image as PILImage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

GCP_PROJECT = "prochazka-ml-playground"
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
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
    whitespace_score: float
    attention_focal_point_score: float
    attention_hierarchy_score: float
    attention_contrast_score: float
    background_blend_risk: str  # low | medium | high
    background_blend_explanation: str | None = None
    crop_recommendation: str | None = None
    # Logo — Gemini determines the advertiser/retailer logo, ignoring product brand marks on the item itself
    advertiser_logo_text: str | None = None  # visible text of the advertiser/retailer logo, e.g. "sportano.bg"
    advertiser_logo_position: str | None = None  # top-left | top-right | bottom-left | bottom-right | top-center | bottom-center | none
    # Score justifications — one sentence explaining WHY this image got each score
    justification_attention: str | None = None
    justification_message_clarity: str | None = None
    justification_branding: str | None = None
    justification_offer_strength: str | None = None
    justification_product_presentation: str | None = None
    justification_readability: str
    justification_whitespace: str | None = None
    improvements_to_perfect_score: list[str]  # 3-5 concrete actions the advertiser could take to reach a perfect overall score
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
    # Safe-zone compliance
    safe_zone_score: float  # 0.0–1.0: how well critical elements stay within the platform safe zone
    safe_zone_elements_at_risk: list[str]  # list of elements at risk, e.g. ["headline near top edge", "price in bottom 35%"]
    justification_safe_zone: str  # actionable tip to improve safe zone compliance
    # DPA category (internal — not surfaced in UI)
    dpa_category: str  # "Luxury / Editorial" | "Lifestyle Fashion" | "Fast Fashion / Volume" | "Sport & Performance" | "Marketplace / Mass Retail" | "Home, Furniture & Electronics" | "Beauty & Wellness" | "Agency / Aggregator"


def _rgb_to_color_name(r: int, g: int, b: int) -> str:
    """Convert an RGB colour to a human-readable name using HSL."""
    r_, g_, b_ = r / 255.0, g / 255.0, b / 255.0
    max_c, min_c = max(r_, g_, b_), min(r_, g_, b_)
    l = (max_c + min_c) / 2.0
    d = max_c - min_c

    # Clamp near-extremes to neutral early — the saturation formula amplifies
    # tiny channel differences near pure white/black into misleading hue values.
    if d < 0.06 or l > 0.93 or l < 0.07:
        if l > 0.88: return "white"
        if l > 0.68: return "light grey"
        if l > 0.38: return "grey"
        if l > 0.15: return "dark grey"
        return "black"

    s = d / (2.0 - max_c - min_c) if l > 0.5 else d / (max_c + min_c)
    if s < 0.12:
        if l > 0.88: return "white"
        if l > 0.68: return "light grey"
        if l > 0.38: return "grey"
        if l > 0.15: return "dark grey"
        return "black"

    if max_c == r_:
        h = ((g_ - b_) / d) % 6
    elif max_c == g_:
        h = (b_ - r_) / d + 2
    else:
        h = (r_ - g_) / d + 4
    h = (h * 60) % 360

    if h < 20:    hue = "red"
    elif h < 45:  hue = "orange"
    elif h < 70:  hue = "yellow"
    elif h < 150: hue = "green"
    elif h < 200: hue = "cyan"
    elif h < 280: hue = "blue"
    elif h < 320: hue = "purple"
    elif h < 345: hue = "pink"
    else:         hue = "red"

    if l > 0.72: return f"light {hue}"
    if l < 0.28: return f"dark {hue}"
    return hue


def _relative_luminance(r: int, g: int, b: int) -> float:
    """WCAG 2.0 relative luminance of an sRGB colour."""
    def _lin(c: float) -> float:
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _wcag_contrast_ratio(rgb1: tuple, rgb2: tuple) -> float:
    """WCAG 2.0 contrast ratio between two RGB tuples. Range: 1–21."""
    l1 = _relative_luminance(*rgb1)
    l2 = _relative_luminance(*rgb2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def _wcag_level(ratio: float) -> str:
    if ratio >= 7.0:
        return "AAA"
    if ratio >= 4.5:
        return "AA"
    if ratio >= 3.0:
        return "AA Large"
    return "Fail"


def _sample_region_contrast(img: PILImage.Image, x1: int, y1: int, x2: int, y2: int) -> float | None:
    """
    Compute the WCAG 2.0 contrast ratio for a single bounding box by:
    - fg  = average of darkest 20 % of pixels inside the box
    - bg  = trimmed mean of the border ring (middle 70 % by brightness)
    Returns None when the sample is unreliable (complex background or fg ≈ bg).
    """
    iw, ih = img.width, img.height
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:
        return None

    # Foreground: darkest 20 % of pixels inside the region
    inner = img.crop((x1, y1, x2, y2))
    pixels = list(inner.getdata())
    if not pixels:
        return None
    brightness = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in pixels]
    cutoff = sorted(brightness)[max(0, int(len(brightness) * 0.2) - 1)]
    ink = [(r, g, b) for (r, g, b), br in zip(pixels, brightness) if br <= cutoff]
    if not ink:
        return None
    fg = tuple(sum(c[i] for c in ink) // len(ink) for i in range(3))

    # Background: border ring, trimmed middle 70 % by brightness
    pad_h = max(4, int(bh * 0.6))
    pad_w = max(4, int(bw * 0.2))
    bx1 = max(0, x1 - pad_w); by1 = max(0, y1 - pad_h)
    bx2 = min(iw, x2 + pad_w); by2 = min(ih, y2 + pad_h)
    outer = img.crop((bx1, by1, bx2, by2))
    outer_pixels = list(outer.getdata())
    ow = outer.width
    ix1, iy1, ix2, iy2 = x1 - bx1, y1 - by1, x1 - bx1 + bw, y1 - by1 + bh
    ring = [px for idx, px in enumerate(outer_pixels)
            if not (ix1 <= (idx % ow) < ix2 and iy1 <= (idx // ow) < iy2)]
    if not ring:
        return None

    ring_sorted = sorted(ring, key=lambda px: 0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2])
    lo, hi = int(len(ring_sorted) * 0.15), int(len(ring_sorted) * 0.85)
    core = ring_sorted[lo:hi] if hi > lo else ring_sorted
    if not core:
        return None
    core_mean = [sum(c[i] for c in core) / len(core) for i in range(3)]
    core_var = sum(sum((px[i] - core_mean[i]) ** 2 for i in range(3)) for px in core) / (len(core) * 3)
    if core_var > 400:
        return None  # complex/textured background

    bg = tuple(int(m) for m in core_mean)
    ratio = _wcag_contrast_ratio(fg, bg)
    return ratio if ratio >= 1.8 else None


def _logo_internal_contrast(img: PILImage.Image, x1: int, y1: int, x2: int, y2: int) -> float | None:
    """
    Measure logo visibility as the contrast between the lightest 20% and darkest 20%
    of pixels inside the bounding box.  This handles both dark-on-light and light-on-dark logos
    (e.g. white text on a red banner), where border-ring methods break down because the ring
    colour is the same as the logo background.
    """
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:
        return None
    pixels = list(img.crop((x1, y1, x2, y2)).getdata())
    if len(pixels) < 10:
        return None
    brightness = sorted(pixels, key=lambda px: 0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2])
    n = len(brightness)
    k = max(1, n // 5)          # 20 % of pixels
    dark_mean  = tuple(sum(px[i] for px in brightness[:k])  // k for i in range(3))
    light_mean = tuple(sum(px[i] for px in brightness[-k:]) // k for i in range(3))
    ratio = _wcag_contrast_ratio(dark_mean, light_mean)
    return ratio if ratio >= 1.5 else None  # below 1.5 → nearly uniform region, unreliable


def _compute_logo_contrast(image_bytes: bytes, logo_annotations) -> dict:
    """
    Compute WCAG contrast for detected logo bounding boxes.
    Uses internal contrast (lightest vs darkest pixels) so it works for both
    dark-on-light and light-on-dark (e.g. white text on coloured banner) logos.
    Reports the minimum (worst) ratio across all logos.
    """
    empty = {"score": None, "ratio": None, "wcag_level": None, "justification": None}
    if not logo_annotations:
        return empty

    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    iw, ih = img.width, img.height
    ratios = []

    for logo in logo_annotations:
        verts = logo.bounding_poly.vertices
        if not verts:
            continue
        xs = [v.x for v in verts]; ys = [v.y for v in verts]
        x1 = max(0, min(xs)); y1 = max(0, min(ys))
        x2 = min(iw, max(xs)); y2 = min(ih, max(ys))
        r = _logo_internal_contrast(img, x1, y1, x2, y2)
        if r is not None:
            ratios.append(r)

    if not ratios:
        return empty

    min_ratio = min(ratios)
    score = round(min((min_ratio - 1.0) / 6.0, 1.0), 3)
    level = _wcag_level(min_ratio)

    if min_ratio >= 7.0:
        justification = "The logo stands out very clearly from its background."
    elif min_ratio >= 4.5:
        justification = "The logo is clearly visible against its background."
    elif min_ratio >= 3.0:
        justification = "The logo is visible but could have stronger contrast against its background."
    else:
        justification = "The logo is hard to distinguish from its background — low visibility."

    return {"score": score, "ratio": round(min_ratio, 2), "wcag_level": level, "justification": justification}



def _compute_text_contrast(image_bytes: bytes, text_annotations, object_bboxes=None) -> dict:
    """
    Sample individual word bounding boxes from Vision API text annotations,
    split pixels into light/dark halves, compute per-region WCAG 2.0 contrast ratios,
    then score by the 25th-percentile ratio (conservative: 75 % of text regions are at
    least this good).  Also returns min/max ratios and a human-readable justification.
    """
    if not text_annotations or len(text_annotations) < 2:
        return {"score": None, "ratio": None, "ratio_min": None, "ratio_max": None,
                "wcag_level": None, "pass_count": None, "total_count": None, "justification": None}

    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    iw, ih = img.width, img.height
    ratios = []
    ratio_texts = []  # (ratio, fg_name, bg_name) triples

    # text_annotations[0] is the full-page block; [1:] are individual words
    for ann in text_annotations[1:40]:
        verts = ann.bounding_poly.vertices
        if not verts:
            continue
        xs = [v.x for v in verts]
        ys = [v.y for v in verts]
        x1, y1 = max(0, min(xs)), max(0, min(ys))
        x2, y2 = min(iw, max(xs)), min(ih, max(ys))
        if (x2 - x1) < 3 or (y2 - y1) < 3:
            continue

        bw, bh = x2 - x1, y2 - y1
        # Skip single characters and tiny fragments — too noisy
        if bw < 8 or bh < 8 or (bw < 15 and bh < 15):
            continue

        # Skip text that sits inside a detected object — likely product label, not ad copy
        if object_bboxes:
            tx1, ty1, tx2, ty2 = x1 / iw, y1 / ih, x2 / iw, y2 / ih  # normalize to 0-1
            text_area = (tx2 - tx1) * (ty2 - ty1)
            is_product_text = False
            if text_area > 0:
                for (ox1, oy1, ox2, oy2) in object_bboxes:
                    # Expand object bbox by 10% to catch labels near product edges
                    margin_x = (ox2 - ox1) * 0.10
                    margin_y = (oy2 - oy1) * 0.10
                    eox1, eoy1 = ox1 - margin_x, oy1 - margin_y
                    eox2, eoy2 = ox2 + margin_x, oy2 + margin_y
                    inter_w = max(0, min(tx2, eox2) - max(tx1, eox1))
                    inter_h = max(0, min(ty2, eoy2) - max(ty1, eoy1))
                    if (inter_w * inter_h) / text_area > 0.50:
                        is_product_text = True
                        break
            if is_product_text:
                continue

        # ── Text colour: darkest 20 % of pixels inside the bbox ──────────────
        text_region = img.crop((x1, y1, x2, y2))
        text_pixels = list(text_region.getdata())
        if not text_pixels:
            continue
        text_brightness = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in text_pixels]
        cutoff = sorted(text_brightness)[max(0, int(len(text_brightness) * 0.2) - 1)]
        ink = [(r, g, b) for (r, g, b), br in zip(text_pixels, text_brightness) if br <= cutoff]
        if not ink:
            continue
        fg = tuple(sum(c[i] for c in ink) // len(ink) for i in range(3))

        # ── Background colour: border ring around the bbox ────────────────────
        pad_h = max(4, int(bh * 0.6))
        pad_w = max(4, int(bw * 0.2))
        bx1 = max(0, x1 - pad_w)
        by1 = max(0, y1 - pad_h)
        bx2 = min(iw, x2 + pad_w)
        by2 = min(ih, y2 + pad_h)
        outer = img.crop((bx1, by1, bx2, by2))
        outer_pixels = list(outer.getdata())
        ow, oh = outer.width, outer.height
        # Keep only pixels in the border ring (exclude the inner bbox area)
        inner_x1 = x1 - bx1
        inner_y1 = y1 - by1
        inner_x2 = inner_x1 + bw
        inner_y2 = inner_y1 + bh
        bg_pixels = [
            px for idx, px in enumerate(outer_pixels)
            if not (inner_x1 <= (idx % ow) < inner_x2 and inner_y1 <= (idx // ow) < inner_y2)
        ]
        if not bg_pixels:
            continue

        # Use the middle 70 % of background pixels sorted by brightness.
        # This trims outlier colour bleed from nearby elements (e.g. a red badge next
        # to price text on white) while still rejecting genuinely complex / textured
        # photo backgrounds where most pixels are varied.
        bp_sorted = sorted(bg_pixels, key=lambda px: 0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2])
        lo = int(len(bp_sorted) * 0.15)
        hi = int(len(bp_sorted) * 0.85)
        core = bp_sorted[lo:hi] if hi > lo else bp_sorted
        if not core:
            continue
        core_mean = [sum(c[i] for c in core) / len(core) for i in range(3)]
        core_variance = sum(
            sum((px[i] - core_mean[i]) ** 2 for i in range(3)) for px in core
        ) / (len(core) * 3)

        # High core variance → complex photo/texture background → skip (product text)
        if core_variance > 400:
            continue

        bg = tuple(int(m) for m in core_mean)

        # Skip text on light background that overlaps with a detected product
        # object — this is product labelling, not ad copy.  The overlap check
        # uses expanded bboxes so we also catch text just outside the product
        # (e.g. product-card titles on a white panel behind the product).
        bg_brightness = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        if bg_brightness > 200 and object_bboxes:
            tx1n, ty1n = x1 / iw, y1 / ih
            tx2n, ty2n = x2 / iw, y2 / ih
            tx_mid, ty_mid = (tx1n + tx2n) / 2, (ty1n + ty2n) / 2
            _near_product = False
            for (ox1, oy1, ox2, oy2) in object_bboxes:
                # Expand object bbox by 25 % in each direction
                ow, oh = ox2 - ox1, oy2 - oy1
                eox1 = ox1 - ow * 0.25
                eoy1 = oy1 - oh * 0.25
                eox2 = ox2 + ow * 0.25
                eoy2 = oy2 + oh * 0.25
                if eox1 <= tx_mid <= eox2 and eoy1 <= ty_mid <= eoy2:
                    _near_product = True
                    break
            if _near_product:
                continue

        ratio = _wcag_contrast_ratio(fg, bg)

        if ratio >= 1.8:
            # Border-ring method worked — dark-on-light text
            ratios.append(ratio)
            ratio_texts.append((ratio, _rgb_to_color_name(*fg), _rgb_to_color_name(*bg)))
        else:
            # Border-ring failed (fg ≈ bg) but background is uniform — likely
            # light text on dark background.  Fall back to internal contrast
            # (lightest 20% vs darkest 20% inside the bbox, same as logo method).
            internal_ratio = _logo_internal_contrast(img, x1, y1, x2, y2)
            if internal_ratio is not None and internal_ratio >= 2.5:
                # Re-derive fg/bg colour names from brightest/darkest 20%
                pixels_sorted = sorted(
                    text_pixels,
                    key=lambda px: 0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2],
                )
                k = max(1, len(pixels_sorted) // 5)
                dark_mean = tuple(sum(px[i] for px in pixels_sorted[:k]) // k for i in range(3))
                light_mean = tuple(sum(px[i] for px in pixels_sorted[-k:]) // k for i in range(3))
                ratios.append(internal_ratio)
                ratio_texts.append((
                    internal_ratio,
                    _rgb_to_color_name(*light_mean),  # fg = light text
                    _rgb_to_color_name(*dark_mean),    # bg = dark background
                ))

    if not ratios:
        return {"score": None, "ratio": None, "ratio_min": None, "ratio_max": None,
                "wcag_level": None, "pass_count": None, "total_count": None, "justification": None}

    ratios_sorted = sorted(ratios)
    n = len(ratios_sorted)
    p25_ratio = ratios_sorted[max(0, int(n * 0.25) - 1)]  # 25th percentile (conservative)
    min_ratio = ratios_sorted[0]
    max_ratio = ratios_sorted[-1]
    pass_aa = sum(1 for r in ratios if r >= 4.5)

    # Score by 25th-percentile ratio, calibrated so that:
    #   AA pass (ratio ≥ 4.5) → score ≥ 0.81 (green zone)
    #   AA Large (ratio ≥ 3.0) → score ≈ 0.54 (orange zone)
    #   Fail (ratio < 3.0) → score < 0.50 (red zone)
    #   AAA (ratio ≥ 7.0) → score = 1.0
    if p25_ratio >= 4.5:
        # AA to AAA: 0.81 → 1.0
        score = round(0.81 + 0.19 * min((p25_ratio - 4.5) / 2.5, 1.0), 3)
    elif p25_ratio >= 3.0:
        # AA Large to AA: 0.51 → 0.80
        score = round(0.51 + 0.29 * (p25_ratio - 3.0) / 1.5, 3)
    else:
        # Fail: 0.0 → 0.50
        score = round(0.50 * min(p25_ratio / 3.0, 1.0), 3)

    # Human-readable justification — describe by fg/bg color pair, no ratio numbers
    fail = n - pass_aa
    failing_pairs = sorted(
        [(r, fg, bg) for r, fg, bg in ratio_texts if r < 4.5],
        key=lambda x: x[0]  # worst first
    )
    seen, failing_labels = set(), []
    for _, fg_name, bg_name in failing_pairs:
        key = (fg_name, bg_name)
        if key not in seen:
            seen.add(key)
            failing_labels.append(f"{fg_name} text on {bg_name} background")
        if len(failing_labels) >= 3:
            break

    if min_ratio >= 7.0:
        justification = "All text elements are very easy to read against their backgrounds."
    elif min_ratio >= 4.5:
        justification = "All text elements stand out clearly from their backgrounds."
    elif pass_aa == 0 and p25_ratio < 3.0:
        pairs_str = "; ".join(failing_labels) if failing_labels else "various combinations"
        justification = f"Text is hard to read throughout — e.g. {pairs_str}."
    elif pass_aa == 0:
        # All below AA but above AA Large — moderate issue
        pairs_str = "; ".join(failing_labels) if failing_labels else "various combinations"
        justification = f"Some text could use more contrast — e.g. {pairs_str}."
    else:
        pairs_str = "; ".join(failing_labels) if failing_labels else f"{fail} combinations"
        verb = "could" if len(failing_labels) == 1 else "could"
        justification = f"Most text is legible, but {pairs_str} could benefit from stronger contrast."

    return {
        "score": score,
        "ratio": round(p25_ratio, 2),
        "ratio_min": round(min_ratio, 2),
        "ratio_max": round(max_ratio, 2),
        "wcag_level": _wcag_level(p25_ratio),
        "pass_count": pass_aa,
        "total_count": n,
        "justification": justification,
    }


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
        "detected_logos": [],
        "text_contrast": {"score": None, "ratio": None, "ratio_min": None, "ratio_max": None,
                         "wcag_level": None, "pass_count": None, "total_count": None, "justification": None},
        "logo_contrast": {"score": None, "ratio": None, "wcag_level": None, "justification": None},
    }

    # Extract object bboxes (normalized 0-1) for product text filtering
    obj_bboxes = []
    if r.localized_object_annotations:
        for obj in r.localized_object_annotations:
            nv = obj.bounding_poly.normalized_vertices
            if nv:
                oxs = [p.x for p in nv]
                oys = [p.y for p in nv]
                obj_bboxes.append((min(oxs), min(oys), max(oxs), max(oys)))

    # Pass raw annotations for safe zone checks downstream
    result["_raw_text_annotations"] = r.text_annotations or []
    result["_raw_logo_annotations"] = r.logo_annotations or []

    if r.text_annotations:
        result["ocr_text"] = r.text_annotations[0].description.strip()
        result["text_contrast"] = _compute_text_contrast(image_bytes, r.text_annotations, obj_bboxes or None)

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
        img = PILImage.open(io.BytesIO(image_bytes))
        w, h = img.width, img.height
        logo_hints = []
        for logo in r.logo_annotations:
            verts = logo.bounding_poly.vertices
            if verts and w > 0 and h > 0:
                cx = sum(v.x for v in verts) / len(verts) / w
                cy = sum(v.y for v in verts) / len(verts) / h
                v_pos = "top" if cy < 0.4 else ("bottom" if cy > 0.6 else "middle")
                h_pos = "left" if cx < 0.4 else ("right" if cx > 0.6 else "center")
                logo_hints.append(f"{logo.description} at {v_pos}-{h_pos} (confidence {logo.score:.2f})")
            else:
                logo_hints.append(f"{logo.description} (position unknown)")
        result["detected_logos"] = logo_hints
        result["logo_contrast"] = _compute_logo_contrast(image_bytes, r.logo_annotations)
        # Fall back: use first logo for legacy logo_position field
        first = r.logo_annotations[0]
        verts = first.bounding_poly.vertices
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
    client: genai.Client,
) -> ImageAnalysis:
    prompt = USER_PROMPT_TEMPLATE

    response = _gemini_generate(
        client,
        model=GEMINI_MODEL,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ImageAnalysis,
            temperature=0.0,
            top_k=1,
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
        _t0 = time.monotonic()
        result: dict = {"analysis_error": None}

        try:
            image_bytes = _fetch_image(url)
        except Exception as e:
            result["analysis_error"] = f"fetch_failed: {e}"
            return result

        _t_fetch = time.monotonic()
        logger.info("Image fetch took %.1fs (%d bytes)", _t_fetch - _t0, len(image_bytes))
        mime_type = _get_mime_type(image_bytes)

        # Run Vision API and Gemini in parallel — Gemini no longer depends on Vision output
        _empty_vision = {
            "ocr_text": "",
            "dominant_colors": [],
            "product_coverage_pct": None,
            "logo_detected": False,
            "logo_position": None,
            "detected_logos": [],
            "text_contrast": {"score": None, "ratio": None, "ratio_min": None, "ratio_max": None,
                              "wcag_level": None, "pass_count": None, "total_count": None, "justification": None},
            "logo_contrast": {"score": None, "ratio": None, "wcag_level": None, "justification": None},
        }

        with ThreadPoolExecutor(max_workers=2) as pool:
            vision_future = pool.submit(_vision_api_analyze, image_bytes, self._credentials)
            gemini_future = pool.submit(_gemini_analyze, image_bytes, mime_type, self._client)

        try:
            vision_data = vision_future.result()
        except Exception as e:
            logger.warning("Vision API failed for %s: %s", url, e)
            vision_data = _empty_vision
        _t_vision = time.monotonic()
        logger.info("Vision API took %.1fs", _t_vision - _t_fetch)

        result["img_product_coverage_pct"] = vision_data["product_coverage_pct"]
        result["img_logo_detected"] = vision_data["logo_detected"]
        result["img_logo_position"] = vision_data["logo_position"]

        try:
            analysis = gemini_future.result()
        except Exception as e:
            result["analysis_error"] = f"gemini_failed: {e}"
            return result
        _t_gemini = time.monotonic()
        logger.info("Gemini took %.1fs (wall from fetch end)", _t_gemini - _t_fetch)
        logger.info("Total analysis time: %.1fs", _t_gemini - _t0)

        result["img_visual_hierarchy"] = analysis.visual_hierarchy_first_element
        result["img_text_readability_mobile"] = analysis.text_readability_mobile_score
        tc = vision_data["text_contrast"]
        result["img_text_contrast_bg"] = tc["score"]
        result["img_text_contrast_ratio"] = tc["ratio"]          # 25th-pct ratio
        result["img_text_contrast_ratio_min"] = tc["ratio_min"]
        result["img_text_contrast_ratio_max"] = tc["ratio_max"]
        result["img_text_contrast_wcag"] = tc["wcag_level"]
        result["img_text_contrast_pass_count"] = tc["pass_count"]
        result["img_text_contrast_total_count"] = tc["total_count"]
        result["img_text_contrast_justification"] = tc["justification"]
        result["img_whitespace_score"] = analysis.whitespace_score
        result["img_attention_focal_point_score"] = analysis.attention_focal_point_score
        result["img_attention_hierarchy_score"] = analysis.attention_hierarchy_score
        result["img_attention_contrast_score"] = analysis.attention_contrast_score
        result["img_bg_blend_risk"] = analysis.background_blend_risk
        result["img_bg_blend_explanation"] = analysis.background_blend_explanation
        result["img_crop_recommendation"] = analysis.crop_recommendation
        # Use Gemini's logo identification as the authoritative source
        result["img_logo_detected"] = bool(analysis.advertiser_logo_text)
        result["img_logo_position"] = analysis.advertiser_logo_position
        result["img_logo_text"] = analysis.advertiser_logo_text
        lc = vision_data.get("logo_contrast", {})
        result["img_logo_contrast_score"] = lc.get("score")
        result["img_logo_contrast_ratio"] = lc.get("ratio")
        result["img_logo_contrast_wcag"] = lc.get("wcag_level")
        result["img_logo_contrast_justification"] = lc.get("justification")
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
        result["img_justification_attention"] = analysis.justification_attention
        result["img_justification_message_clarity"] = analysis.justification_message_clarity
        result["img_justification_branding"] = analysis.justification_branding
        result["img_justification_offer_strength"] = analysis.justification_offer_strength
        result["img_justification_product_presentation"] = analysis.justification_product_presentation
        result["img_justification_readability_mobile"] = analysis.justification_readability
        readability_parts = [s for s in [analysis.justification_readability, tc.get("justification")] if s]
        result["img_justification_readability"] = "; ".join(readability_parts) if readability_parts else None
        result["img_justification_whitespace"] = analysis.justification_whitespace
        result["img_improvements"] = analysis.improvements_to_perfect_score
        result["img_dpa_category"] = analysis.dpa_category

        # --- Aspect ratio detection ---
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(image_bytes))
        w, h = img.width, img.height
        aspect = w / h if h else 1.0
        if abs(aspect - 9 / 16) < 0.05:
            detected_format = "9:16"
        elif abs(aspect - 4 / 5) < 0.05:
            detected_format = "4:5"
        elif abs(aspect - 1.0) < 0.05:
            detected_format = "1:1"
        else:
            detected_format = f"{w}:{h}"
        result["img_ad_format"] = detected_format
        result["img_dimensions"] = f"{w}x{h}"

        # Safe zone — merge Gemini risks with Vision API-detected risks.
        # Vision API gives us precise bounding boxes for text and logos;
        # we programmatically check if they fall in format danger zones.
        _safe_risks = list(analysis.safe_zone_elements_at_risk)
        _existing_risk_lower = {r.lower() for r in _safe_risks}

        # Define danger zones as (top_frac, bottom_frac) of image height
        if detected_format == "9:16":
            _dz_top, _dz_bottom = 0.14, 0.65  # top 14%, bottom 35%
        elif detected_format in ("4:5", "1:1"):
            _dz_top, _dz_bottom = 0.0, 0.90   # bottom 10%
        else:
            _dz_top, _dz_bottom = 0.0, 0.90

        # Check Vision API text annotations for elements in danger zones.
        # Cluster nearby words into logical groups (within 5% vertical distance)
        # to avoid counting each word as a separate risk.
        _text_anns = vision_data.get("_raw_text_annotations") or []
        _top_zone_words = []  # (y_center, text)
        _bottom_zone_words = []
        for ann in _text_anns[1:30]:  # skip full-page block
            verts = ann.bounding_poly.vertices
            if not verts:
                continue
            ys = [v.y for v in verts]
            text_str = ann.description.strip()
            if len(text_str) < 2:
                continue
            y_center = ((min(ys) + max(ys)) / 2) / h
            if y_center < _dz_top:
                _top_zone_words.append((y_center, text_str))
            elif y_center > _dz_bottom:
                _bottom_zone_words.append((y_center, text_str))

        def _cluster_words(words, zone_name):
            """Group words within 5% vertical distance into single risks."""
            if not words:
                return []
            words_sorted = sorted(words, key=lambda x: x[0])
            clusters = []
            cur_texts = [words_sorted[0][1]]
            cur_y = words_sorted[0][0]
            for y, txt in words_sorted[1:]:
                if abs(y - cur_y) < 0.05:
                    cur_texts.append(txt)
                else:
                    clusters.append((cur_y, cur_texts))
                    cur_texts = [txt]
                    cur_y = y
            clusters.append((cur_y, cur_texts))
            risks = []
            for y, texts in clusters:
                joined = " ".join(texts)
                if len(joined) > 40:
                    joined = joined[:37] + "..."
                risks.append(f"'{joined}' at {y:.0%} vertical ({zone_name})")
            return risks

        _vision_risks = (
            _cluster_words(_top_zone_words, "profile bar zone")
            + _cluster_words(_bottom_zone_words, "caption/CTA zone")
        )

        # Check logo position
        if vision_data.get("logo_detected") and vision_data.get("_raw_logo_annotations"):
            for logo in vision_data["_raw_logo_annotations"]:
                verts = logo.bounding_poly.vertices
                if not verts:
                    continue
                ys = [v.y for v in verts]
                y_center = ((min(ys) + max(ys)) / 2) / h
                if y_center < _dz_top:
                    _vision_risks.append(f"logo at {y_center:.0%} vertical (profile bar zone)")
                elif y_center > _dz_bottom:
                    _vision_risks.append(f"logo at {y_center:.0%} vertical (caption/CTA zone)")

        # Merge: add Vision risks not already covered by Gemini
        for vr in _vision_risks:
            if not any(vr.lower()[:20] in r.lower() for r in _safe_risks):
                _safe_risks.append(vr)
        _safe_risks = _safe_risks[:6]

        n_risks = len(_safe_risks)
        if detected_format == "9:16":
            _sz_score = max(0.0, 1.0 - n_risks * 0.2)
        else:
            _sz_score = max(0.0, 1.0 - n_risks * 0.15)
        result["img_safe_zone_score"] = round(_sz_score, 2)
        result["img_safe_zone_elements_at_risk"] = _safe_risks
        result["img_justification_safe_zone"] = analysis.justification_safe_zone if _safe_risks == list(analysis.safe_zone_elements_at_risk) else (
            "Elements at risk: " + "; ".join(_safe_risks) + ". Move critical elements toward the vertical center."
            if _safe_risks else analysis.justification_safe_zone
        )

        # --- Per-category weight profiles for the overall creative score ---
        # Order: attention, message_clarity, branding, offer_strength,
        #         product_presentation, readability, emotional_resonance,
        #         aesthetic_craft, simplicity (1 − clutter), safe_zone
        _CATEGORY_WEIGHTS = {
            #                               attn  msg   brand offer  prod  read  emot  aesth simpl safe
            "Luxury / Editorial":           (1.0, 0.7, 1.0, 0.3, 1.0, 1.0, 1.5, 1.5, 1.2, 0.8),
            "Lifestyle Fashion":            (1.0, 0.8, 1.0, 0.6, 1.0, 1.0, 1.3, 1.3, 1.0, 0.8),
            "Fast Fashion / Volume":        (1.0, 1.0, 0.8, 1.3, 1.0, 1.2, 0.6, 0.6, 0.7, 1.0),
            "Sport & Performance":          (1.2, 1.0, 1.0, 0.8, 1.2, 1.0, 1.2, 1.0, 1.0, 0.8),
            "Marketplace / Mass Retail":    (1.0, 1.0, 0.7, 1.5, 1.0, 1.2, 0.4, 0.5, 0.8, 1.0),
            "Home, Furniture & Electronics":(1.0, 1.0, 0.8, 1.2, 1.3, 1.0, 0.7, 0.8, 1.0, 0.8),
            "Beauty & Wellness":            (1.0, 1.0, 1.0, 0.7, 1.2, 1.0, 1.3, 1.3, 1.0, 0.8),
            "Agency / Aggregator":          (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8),
        }
        _DEFAULT_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8)

        # Composite scores (simple means of relevant sub-scores)
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
        result["img_branding_score"] = _mean([
            analysis.branding_logo_visibility_score,
            analysis.branding_distinctiveness_score,
        ])
        base_offer = _mean(
            [
                analysis.offer_prominence_score,
                analysis.offer_relevance_score,
            ]
        )
        # Price-anchoring modifier for price-driven categories.
        # In Marketplace / Fast Fashion / Home categories, price is the
        # primary conversion driver.  The Gemini-scored prominence/relevance
        # doesn't fully capture the *structural* impact of price anchoring
        # tactics (before/after, discount badge) vs. missing price entirely.
        _PRICE_DRIVEN = {
            "Marketplace / Mass Retail", "Fast Fashion / Volume",
            "Home, Furniture & Electronics",
        }
        if analysis.dpa_category in _PRICE_DRIVEN:
            anchoring_bonus = 0.0
            if analysis.has_before_after_price:
                anchoring_bonus += 0.10       # strongest: crossed-out → sale price
            if analysis.has_discount_pct:
                anchoring_bonus += 0.05       # "−30%" badge adds urgency
            if not analysis.has_price:
                anchoring_bonus -= 0.25       # no visible price at all → severe penalty
            base_offer = max(0.0, min(1.0, base_offer + anchoring_bonus))
        result["img_offer_strength_score"] = base_offer
        _product_parts = [
            analysis.product_clarity_score,
            analysis.product_context_fit_score,
        ]
        if analysis.multi_view_has_multiple_shots:
            _product_parts.extend([
                analysis.product_multi_view_complementarity_score,
                analysis.product_multi_view_clarity_score,
                analysis.product_multi_view_layout_efficiency_score,
            ])
        result["img_product_presentation_score"] = _mean(_product_parts)
        result["img_readability_score"] = _mean(
            [
                tc["score"],
                analysis.text_readability_mobile_score,
            ]
        )
        # ── Category-aware overall score ──────────────────────────────────
        #
        # Weighted geometric mean: naturally multiplicative — one bad
        # score drags the total down without ever pushing the overall
        # below the lowest individual dimension.
        #
        # RELEVANCE GATE (category weight < 0.6 → excluded)
        #   Irrelevant dimensions (e.g. emotional resonance for
        #   Marketplace) are dropped so they can't distort the score.
        _RELEVANCE_GATE = 0.6

        # Combine clutter and whitespace into a single simplicity score
        # to guard against contradictory Gemini outputs (e.g. clutter=0.9
        # but whitespace=0.8 on the same image).
        _simplicity = _mean([1.0 - analysis.clutter_score, analysis.whitespace_score])
        result["img_simplicity_score"] = _simplicity

        # Safe zone weight boost for 9:16 — the most overlay-heavy format.
        # Key category elements in danger zones are penalised more harshly.
        # Use the merged risk list (Vision API + Gemini), not just Gemini's.
        _n_safe_risks = len(result.get("img_safe_zone_elements_at_risk", []))
        _safe_zone_weight_boost = 0.0
        if detected_format == "9:16":
            if _n_safe_risks >= 3:
                _safe_zone_weight_boost = 1.2   # many critical elements at risk
            elif _n_safe_risks >= 1:
                _safe_zone_weight_boost = 0.6   # some elements at risk

        _overall_scores = [
            result["img_attention_score"],
            result["img_message_clarity_score_v2"],
            result["img_branding_score"],
            result["img_offer_strength_score"],
            result["img_product_presentation_score"],
            result["img_readability_score"],
            analysis.emotional_resonance_score,
            analysis.aesthetic_craft_score,
            _simplicity,
            result["img_safe_zone_score"],
        ]
        _weights = list(_CATEGORY_WEIGHTS.get(analysis.dpa_category, _DEFAULT_WEIGHTS))
        _weights[9] += _safe_zone_weight_boost  # boost safe_zone weight for 9:16
        # Keep only dimensions whose category weight meets the relevance gate
        _pairs = [
            (s, w)
            for s, w in zip(_overall_scores, _weights)
            if s is not None and w >= _RELEVANCE_GATE
        ]
        if _pairs:
            _eps = 1e-6
            total_w = sum(w for _, w in _pairs)

            # Weighted geometric mean
            log_sum = sum(
                w * math.log(max(s, _eps)) for s, w in _pairs
            ) / total_w
            base_score = math.exp(log_sum)

            # ── Creative-effort penalty ──────────────────────────────
            # Ads missing fundamental creative elements (logo, headline,
            # CTA) are bare product listings with near-zero creative
            # effort.  Even if individual dimensions score well (the
            # image is "clear" because there's nothing on it), the
            # overall score must reflect the lack of craft.
            _missing = 0
            if not result.get("img_logo_detected"):
                _missing += 1
            if analysis.message_headline_clarity_score <= 0.1:
                _missing += 1
            if analysis.message_cta_clarity_score <= 0.1:
                _missing += 1
            if analysis.aesthetic_craft_score <= 0.3:
                _missing += 1
            # Each missing fundamental reduces the score by 15 %
            if _missing >= 2:
                _effort_penalty = max(0.4, 1.0 - _missing * 0.15)
                base_score *= _effort_penalty

            result["img_overall_creative_score"] = base_score
        else:
            result["img_overall_creative_score"] = 0.0

        return result

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        image_url_col: str = "image_url",
        max_workers: int = MAX_WORKERS,
    ) -> pd.DataFrame:
        import pandas as pd  # noqa: F811 — lazy import to avoid cold-start penalty

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
