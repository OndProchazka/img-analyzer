# Product Image Analyzer

Scores e-commerce ad creatives using **Google Cloud Vision API** and **Gemini** (gemini-3.1-flash-lite-preview). Deployed on **Cloud Run** with a web UI and a BigQuery Remote Function endpoint.

**Live app:** https://img-analyzer-529403971404.europe-central2.run.app/

## How it works

1. User provides a product image URL (via the web UI or the BigQuery remote function).
2. The image is downloaded and sent through two parallel analysis pipelines:
   - **Google Cloud Vision** — dominant colors, text detection (OCR), logo detection, safe-search, object localization.
   - **Gemini** — structured analysis against a detailed scoring rubric (Pydantic schema), returning 30+ dimensions.
3. Vision API results (WCAG contrast ratios, text/background color pairs, product coverage %) are merged with Gemini's assessment.
4. The combined result is returned as JSON — the web UI renders it as score bars, badges, and improvement suggestions.

## Scoring dimensions

| Category | Dimensions |
|----------|-----------|
| **Visual quality** | Attention (focal point, hierarchy, contrast), readability, whitespace, text contrast vs background (WCAG 2.0), safe-zone compliance |
| **Branding** | Logo visibility, logo contrast, brand distinctiveness |
| **Offer & anchoring** | Price detection, discount %, before/after price, coupon, offer prominence, offer relevance |
| **Product presentation** | Clarity, context fit, multi-view detection (shots count, layout type), product coverage %, background blend risk |
| **Message clarity** | Headline clarity, CTA clarity, text density, language clarity |
| **Craft** | Emotional resonance, aesthetic craft, clutter score |
| **Meta safe zones** | Per-format (9:16, 4:5, 1:1) danger zone analysis for UI overlays |
| **DPA category** | Auto-classification into 8 categories (Luxury, Lifestyle Fashion, Fast Fashion, Sport, Marketplace, Home/Electronics, Beauty, Agency) with category-aware scoring adjustments |

## Project structure

```
├── Dockerfile                 # Python 3.12-slim + uv
├── pyproject.toml             # Dependencies and tool config
├── uv.lock                    # Locked dependencies
└── src/
    ├── server.py              # FastAPI app (web UI + /invoke + BigQuery remote function)
    ├── web/
    │   └── index.html         # Single-page frontend (vanilla JS)
    └── product_categorizer/
        ├── __init__.py
        ├── image_analyzer.py  # Core analysis: Vision API + Gemini + WCAG contrast
        ├── google_taxonomy.json
        └── prompts/
            ├── analysis_system.md       # Main scoring rubric & DPA category framework
            ├── analysis_user.md         # User prompt template
            ├── taxonomy_keyword.md      # Google product taxonomy prompts
            ├── taxonomy_leaf_first.md
            ├── taxonomy_two_step_l1.md
            └── taxonomy_two_step_leaf.md
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET /` | Web UI | Serves the single-page analyzer interface |
| `POST /invoke` | Single image analysis | `{"parameters": {"image_url": "..."}}` → full analysis JSON |
| `POST /bq` | BigQuery Remote Function | `{"calls": [["url1"], ["url2"]]}` → `{"replies": [score1, score2]}` |
| `POST /` | BigQuery Remote Function (alt) | Same as `/bq` — BQ remote functions only support root URL |

## Local development

```bash
# Prerequisites: Python 3.12+, uv, GCP credentials with access to prochazka-ml-playground
uv sync
uv run python src/server.py
# Open http://localhost:8080
```

## Deployment

Built and deployed via Cloud Build to Cloud Run (`europe-central2`):

```bash
gcloud builds submit \
  --tag europe-central2-docker.pkg.dev/prochazka-ml-playground/ml-images/img-analyzer:latest \
  --project prochazka-ml-playground

gcloud run deploy img-analyzer \
  --image europe-central2-docker.pkg.dev/prochazka-ml-playground/ml-images/img-analyzer:latest \
  --region europe-central2 \
  --project prochazka-ml-playground \
  --allow-unauthenticated
```

## GCP dependencies

- **Cloud Vision API** — text detection, logo detection, dominant colors, object localization
- **Vertex AI / Gemini** — structured image analysis via `google-genai` SDK
- **Cloud Run** — serverless hosting
- **Artifact Registry** — container image storage (`europe-central2-docker.pkg.dev/prochazka-ml-playground/ml-images`)
