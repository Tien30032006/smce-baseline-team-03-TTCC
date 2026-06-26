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
        <li><strong>Thành viên:</strong> {cfg.TEAM_MEMBERS}</li>
        <li><strong>GitHub:</strong> <a href="{cfg.GITHUB_REPO}" target="_blank">{cfg.GITHUB_REPO}</a></li>
        <li><strong>Tài nguyên khác:</strong> <a href="{cfg.OTHER_RESOURCE}" target="_blank">{cfg.OTHER_RESOURCE}</a></li>
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


@st.cache_resource(show_spinner="Đang chạy benchmark khởi động (1 ảnh)...")
def _cached_deploy_smoke() -> dict:
    return get_deploy_smoke_benchmark()


def _render_about_tab() -> None:
    st.header("Giới thiệu")
    st.markdown(
        """
        Tab này trình bày giải pháp OCR + trích xuất **brand_name** và **product_name**
        của **Team 03 - TTCC** tại cuộc thi **The 2nd URA Hackathon 2026**.
        """
    )

    st.subheader("1. Thông tin nhóm")
    st.markdown(
        f"""
        | Trường | Nội dung |
        |--------|----------|
        | **Tên nhóm** | {cfg.TEAM_NAME} |
        | **Thành viên** | Huỳnh Tấn Tiến |
        | | Phạm Nguyễn Công Thành |
        | | Nguyễn Trần Lan Châu |
        | | Trương Lê Đan Chi |
        | **GitHub** | [{cfg.GITHUB_REPO}]({cfg.GITHUB_REPO}) |
        """
    )

    st.subheader("2. Bài toán")
    st.markdown(
        """
        Từ **ảnh sản phẩm trên kệ hàng hoặc mạng xã hội**, hệ thống cần trích xuất:

        - **`ocr_text`** — toàn bộ văn bản nhận dạng được từ ảnh
        - **`brand_name`** — tên thương hiệu sản phẩm
        - **`product_name`** — tên hoặc mô tả sản phẩm

        **Công thức tính điểm (private round):**

        `0.4 × F1_brand + 0.35 × (1 − CER) + 0.25 × F1_product`
        """
    )

    st.subheader("3. Pipeline giải pháp")
    st.markdown(
        """
        1. **Tiền xử lý ảnh** — Chuyển về RGB, resize nếu cần, tăng độ tương phản để cải thiện chất lượng OCR
        2. **OCR** — PaddleOCR (PP-OCRv4, hỗ trợ tiếng Việt + tiếng Anh), chạy trên CPU
        3. **Hậu xử lý OCR** — Loại bỏ token trùng lặp, chuẩn hoá Unicode, ghép dòng liên quan
        4. **Trích xuất brand** — Regex kết hợp từ điển thương hiệu; fallback sang fuzzy matching (rapidfuzz)
        5. **Trích xuất product** — Rule-based ưu tiên; fallback sang TF-IDF + Logistic Regression
        6. **Hậu kiểm** — Loại bỏ kết quả có độ tin cậy thấp, trả về chuỗi rỗng nếu không chắc chắn
        """
    )

    st.subheader("4. Điểm nổi bật")
    st.markdown(
        """
        - **Không phụ thuộc GPU** — toàn bộ pipeline chạy trên CPU, phù hợp với môi trường Streamlit Cloud
        - **Hỗ trợ song ngữ** — PaddleOCR nhận dạng cả tiếng Việt lẫn tiếng Anh trong cùng một ảnh
        - **Nhẹ & nhanh** — Brand/product extraction dùng regex + TF-IDF, không cần model nặng
        - **Dễ mở rộng** — Từ điển brand và rule product có thể cập nhật độc lập mà không cần train lại
        """
    )

    st.subheader("5. Công nghệ sử dụng")
    st.markdown(
        """
        | Thành phần | Công nghệ |
        |------------|-----------|
        | OCR | PaddleOCR (PP-OCRv4, vi+en) |
        | Brand extraction | Regex rules + fuzzy matching (rapidfuzz) |
        | Product extraction | Rule-based + TF-IDF / Logistic Regression |
        | Runtime | CPU — Python 3.11 |
        | Demo UI | Streamlit |
        """
    )

    st.subheader("6. Kết quả & đánh giá")
    st.markdown(
        """
        | Metric | Giá trị |
        |--------|---------|
        | F1 brand (local) | — |
        | 1 − CER (local) | — |
        | F1 product (local) | — |
        | **Private score** | — |
        | Latency trung bình / ảnh | — ms |
        | Kích thước product head | — MB |
        """
    )
    st.markdown(
        """
        **Chạy benchmark đầy đủ (local):**

        ```bash
        python scripts/benchmark_solution.py --limit 6
        ```

        Cập nhật `MODEL_PROFILE` trong [`team_config.py`](team_config.py) khi thay đổi OCR hoặc model.
        """
    )

    st.subheader("7. Hạn chế & hướng phát triển")
    st.markdown(
        """
        **Hạn chế hiện tại**
        - Thương hiệu mới chưa có trong từ điển có thể bị bỏ sót
        - Ảnh bị mờ, nghiêng hoặc ánh sáng kém ảnh hưởng đến độ chính xác OCR
        - PaddleOCR cold start lần đầu tốn thời gian tải weights (~vài trăm MB)

        **Hướng phát triển**
        - Fine-tune OCR trên tập dữ liệu sản phẩm bán lẻ Việt Nam
        - Mở rộng từ điển brand tự động từ dữ liệu huấn luyện
        - Thử nghiệm mô hình NER hoặc LLM nhỏ cho trích xuất product
        """
    )

    st.subheader("8. Liên kết")
    links = [
        f"- **Repository:** [{cfg.GITHUB_REPO}]({cfg.GITHUB_REPO})",
        "- **Hướng dẫn cài đặt & deploy:** [README.md](README.md)",
        f"- **Tài nguyên khác:** [{cfg.OTHER_RESOURCE}]({cfg.OTHER_RESOURCE})",
    ]
    streamlit_url = getattr(cfg, "STREAMLIT_APP_URL", "")
    if streamlit_url:
        links.insert(
            1,
            f"- **Live demo (Streamlit Cloud):** [{streamlit_url}]({streamlit_url})",
        )
    st.markdown("\n".join(links))


tab_live, tab_about = st.tabs(["Live test", "Giới thiệu"])

with tab_live:
    _init_live_state()
    st.subheader("Live test")

    profile = _cached_model_profile()
    smoke = _cached_deploy_smoke()
    with st.expander("Thông tin mô hình (kiểm tra nhẹ)", expanded=False):
        st.markdown(
            f"- **Pipeline:** {profile.get('pipeline', '—')}\n"
            f"- **Runtime:** {profile.get('runtime_device', '—')}\n"
            f"- **Product head:** {profile.get('product_head_mb', 0)} MB\n"
            f"- **Ghi chú OCR:** {profile.get('ocr_backend_note', '—')}\n\n"
            f"{profile.get('lightweight_notes', '')}"
        )
        if smoke.get("latency_ms"):
            lat = smoke["latency_ms"]
            st.markdown(
                f"**Benchmark khởi động (1 ảnh):** "
                f"tổng **{lat.get('total_avg', '—')} ms** "
                f"(ocr {lat.get('ocr_avg', '—')} · extract {lat.get('extract_avg', '—')})"
            )
        elif smoke.get("error"):
            st.caption(f"Benchmark khởi động bị bỏ qua: {smoke['error']}")
        st.caption("Báo cáo đầy đủ: `python scripts/benchmark_solution.py --limit 6`")

    uploaded = st.file_uploader(
        "Ảnh sản phẩm",
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
            if st.button("Chạy OCR", type="primary", key="run_ocr_live"):
                with st.spinner("Đang chạy OCR..."):
                    pred = run_predict_with_metrics(img)
                    st.session_state["ocr_text_live"] = pred["ocr_text"]
                    st.session_state["brand_name_live"] = pred["brand_name"]
                    st.session_state["product_name_live"] = pred["product_name"]
                    st.session_state["timing_ms"] = pred.get("timing_ms")

            timing = st.session_state.get("timing_ms")
            if timing:
                t1, t2, t3 = st.columns(3)
                t1.metric("Tổng (ms)", f"{timing['total']:.1f}")
                t2.metric("OCR (ms)", f"{timing['ocr']:.1f}")
                t3.metric("Trích xuất (ms)", f"{timing['extract']:.1f}")

            st.text_area("ocr_text", height=140, key="ocr_text_live")
            st.text_input("brand_name", key="brand_name_live")
            st.text_input("product_name", key="product_name_live")
    else:
        st.session_state["upload_file_id"] = None
        _clear_live_results()

with tab_about:
    _render_about_tab()