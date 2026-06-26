from __future__ import annotations

import os
import re
import time
import traceback
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

from solution.brand_rules import extract_brand_product  # noqa: F401  (re-exported for callers)
from solution.product_model import predict_brand_product, restore_text
from team_config import DEFAULT_MIN_CONF

# --- Cell 1 (Setup) — environment flags that must be set before paddle import.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_onednn", "0")
os.environ.setdefault("DNNL_DEFAULT_FPMATH_MODE", "STRICT")
os.environ.setdefault("PADDLE_DISABLE_MKL", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

# Older Pillow compatibility shim required by some paddleocr versions.
import PIL._util as _pil_util  # noqa: E402

if not hasattr(_pil_util, "is_directory"):
    _pil_util.is_directory = lambda f: os.path.isdir(f)
if not hasattr(_pil_util, "is_path"):
    _pil_util.is_path = lambda f: isinstance(f, (str, bytes, os.PathLike))


# ---------------------------------------------------------------------------
# Image preprocessing (Cell 4 — load_and_prep, adapted: takes an Image, not a path)
# ---------------------------------------------------------------------------

def preprocess(img: Image.Image, max_dim: int = 1280, min_dim: int = 480) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) < min_dim:
        s = min_dim / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    elif max(w, h) > max_dim:
        s = max_dim / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    img = ImageEnhance.Sharpness(img).enhance(1.8)
    img = ImageEnhance.Contrast(img).enhance(1.35)
    return img


# ---------------------------------------------------------------------------
# OCR post-processing (Cell 4 — _fix, _dedup, postprocess_ocr)
# ---------------------------------------------------------------------------

_OCR_FIXES = [
    (r"\b[Vv]inam[i1l][l1]k\b",    "Vinamilk"),
    (r"\b[Cc]anf[uo][ck][oa]\b",   "Canfoco"),
    (r"\bTH.?Tru[e3]\b",           "TH True"),
    (r"\b[Nn]estl[e3é]\b",         "Nestlé"),
    (r"\b[Mm][i1]lo\b",            "Milo"),
    (r"\b[Hh]ighland[s]?\b",       "Highlands"),
    (r"\b[Hh][aà].?[Ll][o0]ng\b",  "Hạ Long"),
    (r"\b[Nn][Aa][Nn]\b",          "NAN"),
    (r"\b[Aa]ptam[i1]l\b",         "Aptamil"),
    (r"\b[Hh][Ii][Pp][Pp]\b",      "HiPP"),
    (r"\b[Vv]issan\b",             "Vissan"),
    (r"(?<=[a-z])0(?=[a-z])",      "o"),
]


def _fix(text: str) -> str:
    for pat, rep in _OCR_FIXES:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text


def _dedup(text: str) -> str:
    toks = text.split()
    if not toks:
        return ""
    out = [toks[0]]
    for t in toks[1:]:
        if t.lower() != out[-1].lower():
            out.append(t)
    return " ".join(out)


def postprocess_ocr(text: str) -> str:
    text = re.sub(r"[\n\t\r]", " ", text)
    text = re.sub(r"[^\w\s\u00C0-\u024F\u1EA0-\u1EF9.,;:!?()/%%\-@#]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _fix(text)
    text = restore_text(text)
    text = _dedup(text)
    return text[:500]


def _is_noise(raw: str) -> bool:
    if not raw:
        return True
    toks = raw.split()
    if len(toks) <= 1 and len(raw) < 4:
        return True
    alpha = sum(1 for t in toks if re.search(r"[a-zA-Z\u00C0-\u024F\u1EA0-\u1EF9]", t))
    return alpha / max(len(toks), 1) < 0.20


# ---------------------------------------------------------------------------
# OCR engine (Cell 4 — PaddleOCR only, matches the notebook exactly)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_paddle_ocr():
    from paddleocr import PaddleOCR

    return PaddleOCR(
        ocr_version="PP-OCRv4",
        use_angle_cls=True,
        lang="vi",
        use_gpu=False,
        show_log=False,
        det_db_score_mode="slow",
        rec_batch_num=8,
    )


def _parse(result, thresh: float = 0.30) -> list[tuple[str, float]]:
    lines = []
    if not result or not result[0]:
        return lines
    for line in result[0]:
        if not line or len(line) < 2:
            continue
        ti = line[1]
        if isinstance(ti, (list, tuple)) and len(ti) >= 2:
            txt, score = str(ti[0]).strip(), float(ti[1])
            if score >= thresh and txt:
                lines.append((txt, score))
    return lines


def _run_paddle_full(img: Image.Image) -> tuple[str, float]:
    try:
        result = get_paddle_ocr().ocr(np.array(img), cls=True)
    except Exception as e:
        print(f"\n[PADDLE OCR ERROR]: {e}")
        traceback.print_exc()
        return "", 0.0
    pairs = _parse(result, thresh=0.30)
    if not pairs:
        return "", 0.0
    raw = " ".join(t for t, _ in pairs)
    mean_score = sum(s for _, s in pairs) / len(pairs)
    return raw, mean_score


def run_ocr_on_image(img: Image.Image, weak_score_thresh: float = 0.55) -> str:
    img = preprocess(img)
    raw, score = _run_paddle_full(img)

    if _is_noise(raw):
        return ""
    return postprocess_ocr(raw)


# ---------------------------------------------------------------------------
# Main entry point (Cell 5, adapted to the template's image-in / dict-out contract)
# ---------------------------------------------------------------------------

def predict_from_text(ocr_text: str) -> tuple[str, str]:
    """Extract brand + product from raw OCR text (no image)."""
    return predict_brand_product(ocr_text or "")


def predict_from_image(
    img: Image.Image,
    min_conf: float = DEFAULT_MIN_CONF,
    *,
    include_timing: bool = True,
) -> dict[str, Any]:
    """
    Main entry point for Streamlit + batch submission.

    Returns dict with keys: ocr_text, brand_name, product_name
    Optional timing_ms: {ocr, extract, total} in milliseconds.
    """
    t0 = time.perf_counter()

    t_ocr = time.perf_counter()
    ocr_text = run_ocr_on_image(img, weak_score_thresh=max(0.30, min(0.9, 1.0 - min_conf)))
    ocr_ms = (time.perf_counter() - t_ocr) * 1000

    t_extract = time.perf_counter()
    brand_name, product_name = predict_brand_product(ocr_text)
    extract_ms = (time.perf_counter() - t_extract) * 1000

    total_ms = (time.perf_counter() - t0) * 1000

    result: dict[str, Any] = {
        "ocr_text": ocr_text,
        "brand_name": brand_name,
        "product_name": product_name,
    }
    if include_timing:
        result["timing_ms"] = {
            "ocr": round(ocr_ms, 1),
            "extract": round(extract_ms, 1),
            "total": round(total_ms, 1),
        }
    return result