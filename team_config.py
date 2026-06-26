"""Team-facing configuration — edit this file after forking the template."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Team identity (required after fork)
TEAM_NAME = "Team 03 - TTCC"
TEAM_MEMBERS = "Huỳnh Tấn Tiến, Phạm Nguyễn Công Thành, Nguyễn Trần Lan Châu, Trương Lê Đan Chi"
GITHUB_REPO = "https://github.com/Tien30032006/Team-03-TTCC"
OTHER_RESOURCE = "https://example.com/other-resource"
STREAMLIT_APP_URL = ""  # e.g. "https://ura-team-abc.streamlit.app" after deploy

# Streamlit page copy
SUBTITLE = (
    "OCR & Product Name Extraction from Social Media Images "
    "by HCMUT URA Research Group"
)
PAGE_TITLE = f"The 2nd URA Hackathon - {TEAM_NAME}"
BROWSER_TITLE = PAGE_TITLE

# Branding assets (replace files under assets/ if needed)
ASSETS_DIR = REPO_ROOT / "assets"
FAVICON = ASSETS_DIR / "kaggle_144224_logos_thumb76_76.png"
LOGO = ASSETS_DIR / "bk_name_en.png"
LOGO_WIDTH = 280

# UI theme
THEME_PRIMARY = "#1565C0"
THEME_PRIMARY_DARK = "#0D47A1"
THEME_BG = "#FFFFFF"
THEME_TEXT = "#1A2B4A"
THEME_MUTED = "#5C6B8A"

# Default inference settings (override inside solution/pipeline.py if needed)
DEFAULT_MIN_CONF = 0.35

# Model footprint (edit when you change OCR / models — benchmark layer reads this)
MODEL_PROFILE: dict[str, str | float | None] = {
    "pipeline": "PaddleOCR (PP-OCRv4, vi) + regex brand rules + TF-IDF product head",
    "runtime_device": "CPU",
    "product_head_mb": None,  # auto-estimate when None
    "ocr_backend_note": "PaddleOCR weights cached on first run (not bundled in repo)",
    "lightweight_notes": (
        "Rule-based brand/product extraction (regex) runs first; falls back to a "
        "lightweight TF-IDF + Logistic Regression head, then fuzzy matching. "
        "OCR runs on CPU only (no GPU dependency)."
    ),
}