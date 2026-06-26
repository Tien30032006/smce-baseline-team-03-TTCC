#!/usr/bin/env python3
"""Streamlit demo shell for URA Hackathon teams — customize team_config.py + solution/."""

from __future__ import annotations

import io

import streamlit as st
from PIL import Image

import team_config as cfg
from shared.benchmark import (
    get_deploy_smoke_benchmark,
    get_model_profile,
    run_predict_with_metrics,
)

APP_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');

:root {{
    --ura-blue: {cfg.THEME_PRIMARY};
    --ura-blue-dark: {cfg.THEME_PRIMARY_DARK};
    --ura-bg: {cfg.THEME_BG};
    --ura-text: {cfg.THEME_TEXT};
    --ura-muted: {cfg.THEME_MUTED};
}}

html, body, .stApp {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
    background-color: var(--ura-bg) !important;
    color: var(--ura-text) !important;
}}

[data-testid="stSidebar"] {{ display: none; }}
[data-testid="collapsedControl"] {{ display: none; }}

[data-testid="stAppViewContainer"] > section > div {{
    padding-top: 1rem;
}}

[data-testid="stImage"]:first-of-type {{
    margin-bottom: 1rem;
}}

[data-testid="stImage"]:first-of-type img {{
    max-height: 72px;
    width: auto;
}}

.app-title,
[data-testid="stMarkdownContainer"] p.app-title {{
    display: block;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 32px !important;
    font-weight: 700 !important;
    color: var(--ura-blue) !important;
    margin: 0 0 0.5rem 0 !important;
    line-height: 1.25 !important;
}}

.app-subtitle {{
    display: block;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: var(--ura-muted) !important;
    margin: 0 0 0.75rem 0 !important;
    line-height: 1.5 !important;
    max-width: 100%;
}}

.app-team-info {{
    margin: 0 0 1.25rem 0;
    padding: 0;
    list-style: none;
}}

.app-team-info li {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    margin: 0 0 0.35rem 0 !important;
    color: var(--ura-text) !important;
}}

.app-team-info li strong {{
    color: var(--ura-blue);
    font-weight: 600;
}}

.app-team-info a {{
    color: var(--ura-blue);
    text-decoration: none;
    font-weight: 500;
}}

.app-team-info a:hover {{
    text-decoration: underline;
}}

[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {{
    font-family: 'Montserrat', sans-serif !important;
    color: var(--ura-blue) !important;
}}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"] {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
}}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
    color: var(--ura-blue) !important;
    border-bottom-color: var(--ura-blue) !important;
}}

.stTabs [data-baseweb="tab-list"] button {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}}

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {{
    background-color: var(--ura-blue) !important;
    border-color: var(--ura-blue) !important;
    color: #FFFFFF !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {{
    background-color: var(--ura-blue-dark) !important;
    border-color: var(--ura-blue-dark) !important;
}}

.stTextInput input,
.stTextArea textarea,
.stTextInput label,
.stTextArea label {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
}}

[data-testid="stFileUploader"] label {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: var(--ura-text) !important;
}}

[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
}}

[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
}}
"""

st.set_page_config(
    page_title=cfg.BROWSER_TITLE,
    page_icon=str(cfg.FAVICON),
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)

st.image(str(cfg.LOGO), width=cfg.LOGO_WIDTH)

st.markdown(
    f'<p class="app-title">{cfg.PAGE_TITLE}</p>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p class="app-subtitle">{cfg.SUBTITLE}</p>',
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <ul class="app-team-info">
        <li><strong>Team Members:</strong> {cfg.TEAM_MEMBERS}</li>
        <li><strong>GitHub Repo:</strong> <a href="{cfg.GITHUB_REPO}" target="_blank">{cfg.GITHUB_REPO}</a></li>
        <li><strong>Other Resource:</strong> <a href="{cfg.OTHER_RESOURCE}" target="_blank">{cfg.OTHER_RESOURCE}</a></li>
    </ul>
    """,
    unsafe_allow_html=True,
)


def _init_live_state() -> None:
    defaults = {
        "ocr_text_live": "",
        "brand_name_live": "",
        "product_name_live": "",
        "upload_file_id": None,
        "timing_ms": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _load_uploaded_image(uploaded) -> Image.Image:
    return Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")


def _clear_live_results() -> None:
    st.session_state["ocr_text_live"] = ""
    st.session_state["brand_name_live"] = ""
    st.session_state["product_name_live"] = ""
    st.session_state["timing_ms"] = None


@st.cache_data(show_spinner=False)
def _cached_model_profile() -> dict:
    return get_model_profile()


@st.cache_resource(show_spinner="Running deploy smoke benchmark (1 image)...")
def _cached_deploy_smoke() -> dict:
    return get_deploy_smoke_benchmark()


def _render_about_tab() -> None:
    st.header("About")
    st.markdown(
        """
        This tab presents the OCR + **brand_name** and **product_name** extraction solution
        of **Team 03 - TTCC** at **The 2nd URA Hackathon 2026**.
        """
    )

    st.subheader("1. Team Information")
    st.markdown(
        f"""
        | Field | Detail |
        |-------|--------|
        | **Team Name** | {cfg.TEAM_NAME} |
        | **Members** | Huynh Tan Tien |
        | | Pham Nguyen Cong Thanh |
        | | Nguyen Tran Lan Chau |
        | | Truong Le Dan Chi |
        | **GitHub** | [{cfg.GITHUB_REPO}]({cfg.GITHUB_REPO}) |
        """
    )

    st.subheader("2. Problem Statement")
    st.markdown(
        """
        Given a **product image from a retail shelf or social media**, the system must extract:

        - **`ocr_text`** — all text recognized from the image
        - **`brand_name`** — the product brand name
        - **`product_name`** — the product name or description

        **Scoring formula (private round):**

        `0.4 × F1_brand + 0.35 × (1 − CER) + 0.25 × F1_product`
        """
    )

    st.subheader("3. Solution Pipeline")
    st.markdown(
        """
        1. **Image pre-processing** — Convert to RGB, resize if needed, enhance contrast to improve OCR quality
        2. **OCR** — PaddleOCR (PP-OCRv4, Vietnamese + English), running on CPU
        3. **OCR post-processing** — Deduplicate tokens, normalize Unicode, merge related lines
        4. **Brand extraction** — Regex combined with a brand dictionary; fallback to fuzzy matching (rapidfuzz)
        5. **Product extraction** — Rule-based first; fallback to TF-IDF + Logistic Regression
        6. **Post-validation** — Drop low-confidence results, return empty string when uncertain
        """
    )

    st.subheader("4. Key Highlights")
    st.markdown(
        """
        - **GPU-free** — the entire pipeline runs on CPU, compatible with Streamlit Cloud
        - **Bilingual support** — PaddleOCR handles Vietnamese and English text in the same image
        - **Lightweight & fast** — brand/product extraction uses regex + TF-IDF, no heavy model required
        - **Easy to extend** — brand dictionary and product rules can be updated independently without retraining
        """
    )

    st.subheader("5. Tech Stack")
    st.markdown(
        """
        | Component | Technology |
        |-----------|------------|
        | OCR | PaddleOCR (PP-OCRv4, vi+en) |
        | Brand extraction | Regex rules + fuzzy matching (rapidfuzz) |
        | Product extraction | Rule-based + TF-IDF / Logistic Regression |
        | Runtime | CPU — Python 3.11 |
        | Demo UI | Streamlit |
        """
    )

    st.subheader("6. Results & Evaluation")
    st.markdown(
        """
        | Metric | Value |
        |--------|-------|
        | F1 brand (local) | — |
        | 1 − CER (local) | — |
        | F1 product (local) | — |
        | **Private score** | — |
        | Avg latency / image | — ms |
        | Product head size | — MB |
        """
    )
    st.markdown(
        """
        **Run full benchmark (local):**

        ```bash
        python scripts/benchmark_solution.py --limit 6
        ```

        Update `MODEL_PROFILE` in [`team_config.py`](team_config.py) whenever you change OCR or models.
        """
    )

    st.subheader("7. Limitations & Future Work")
    st.markdown(
        """
        **Current limitations**
        - Unseen brands not in the dictionary may be missed
        - Blurry, rotated, or poorly lit images reduce OCR accuracy
        - PaddleOCR cold start takes time to download weights (~several hundred MB)

        **Future directions**
        - Fine-tune OCR on Vietnamese retail product datasets
        - Auto-expand the brand dictionary from training data
        - Experiment with a lightweight NER model or small LLM for product extraction
        """
    )

    st.subheader("8. Links")
    links = [
        f"- **Repository:** [{cfg.GITHUB_REPO}]({cfg.GITHUB_REPO})",
        "- **Setup & deploy guide:** [README.md](README.md)",
        f"- **Other resource:** [{cfg.OTHER_RESOURCE}]({cfg.OTHER_RESOURCE})",
    ]
    streamlit_url = getattr(cfg, "STREAMLIT_APP_URL", "")
    if streamlit_url:
        links.insert(
            1,
            f"- **Live demo (Streamlit Cloud):** [{streamlit_url}]({streamlit_url})",
        )
    st.markdown("\n".join(links))


tab_live, tab_about = st.tabs(["Live test", "About"])

with tab_live:
    _init_live_state()
    st.subheader("Live test")

    profile = _cached_model_profile()
    smoke = _cached_deploy_smoke()
    with st.expander("Model footprint (lightweight check)", expanded=False):
        st.markdown(
            f"- **Pipeline:** {profile.get('pipeline', '—')}\n"
            f"- **Runtime:** {profile.get('runtime_device', '—')}\n"
            f"- **Product head:** {profile.get('product_head_mb', 0)} MB\n"
            f"- **OCR note:** {profile.get('ocr_backend_note', '—')}\n\n"
            f"{profile.get('lightweight_notes', '')}"
        )
        if smoke.get("latency_ms"):
            lat = smoke["latency_ms"]
            st.markdown(
                f"**Deploy smoke benchmark (1 image):** "
                f"total **{lat.get('total_avg', '—')} ms** "
                f"(ocr {lat.get('ocr_avg', '—')} · extract {lat.get('extract_avg', '—')})"
            )
        elif smoke.get("error"):
            st.caption(f"Deploy smoke benchmark skipped: {smoke['error']}")
        st.caption("Full report: `python scripts/benchmark_solution.py --limit 6`")

    uploaded = st.file_uploader(
        "Product image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="live_upload",
    )

    if uploaded:
        file_id = f"{uploaded.name}:{uploaded.size}"
        if st.session_state["upload_file_id"] != file_id:
            st.session_state["upload_file_id"] = file_id
            _clear_live_results()

        img = _load_uploaded_image(uploaded)
        col_img, col_result = st.columns(2)

        with col_img:
            st.image(img, width="stretch")

        with col_result:
            if st.button("Run OCR", type="primary", key="run_ocr_live"):
                with st.spinner("Running OCR..."):
                    pred = run_predict_with_metrics(img)
                    st.session_state["ocr_text_live"] = pred["ocr_text"]
                    st.session_state["brand_name_live"] = pred["brand_name"]
                    st.session_state["product_name_live"] = pred["product_name"]
                    st.session_state["timing_ms"] = pred.get("timing_ms")

            timing = st.session_state.get("timing_ms")
            if timing:
                t1, t2, t3 = st.columns(3)
                t1.metric("Total (ms)", f"{timing['total']:.1f}")
                t2.metric("OCR (ms)", f"{timing['ocr']:.1f}")
                t3.metric("Extract (ms)", f"{timing['extract']:.1f}")

            st.text_area("ocr_text", height=140, key="ocr_text_live")
            st.text_input("brand_name", key="brand_name_live")
            st.text_input("product_name", key="product_name_live")
    else:
        st.session_state["upload_file_id"] = None
        _clear_live_results()

with tab_about:
    _render_about_tab()