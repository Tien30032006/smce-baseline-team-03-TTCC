from __future__ import annotations

import re
import re as _re
import time
from collections import Counter
from unicodedata import normalize as _unorm2

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from solution.brand_rules import (
    BRAND_HAS_BUILTIN_LINE,
    BRAND_RULES,
    KNOWN_BRANDS,
    KNOWN_LINES,
    LINE_RULES,
    _find_line,
    _strip_accent,
    extract_brand_product,
    normalize_brand,
    normalize_product,
)

try:
    from rapidfuzz import fuzz, process
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


# ---------------------------------------------------------------------------
# TF-IDF NamePredictor (Cell 3b)
# ---------------------------------------------------------------------------

def _derive_brand_line(row) -> pd.Series:
    brand, _ = extract_brand_product(row["ocr_text"])
    line = ""
    if brand and brand not in BRAND_HAS_BUILTIN_LINE:
        combined = f'{row["ocr_text"]} {row["product_name"]}'
        line = _find_line(brand, combined.lower(), _strip_accent(combined))
    return pd.Series({"brand_derived": brand, "line_derived": line})


class NamePredictor:
    """Two-stage TF-IDF classifier: has-a-name? -> which name?"""

    def __init__(self, prob_threshold: float = 0.45, max_features: int = 15000, min_class_count: int = 2):
        self.prob_threshold = prob_threshold
        self.max_features = max_features
        self.min_class_count = min_class_count
        self._has_clf = None
        self._name_clf = None

    def fit(self, texts, labels) -> "NamePredictor":
        texts = pd.Series(texts).astype(str).str.strip()
        labels = pd.Series(labels).astype(str).str.strip()
        tf = dict(analyzer="char_wb", ngram_range=(2, 4),
                  max_features=self.max_features, min_df=2, sublinear_tf=True)
        lr = dict(max_iter=500, class_weight="balanced", C=2.0)

        self._has_clf = Pipeline([("t", TfidfVectorizer(**tf)),
                                   ("c", LogisticRegression(**lr))])
        self._has_clf.fit(texts, (labels != "").astype(int))

        pos = pd.DataFrame({"text": texts, "label": labels})
        pos = pos[(pos["text"] != "") & (pos["label"] != "")]
        keep = pos["label"].value_counts()
        pos = pos[pos["label"].isin(keep[keep >= self.min_class_count].index)]

        self._name_clf = None
        if len(pos) and pos["label"].nunique() >= 2:
            self._name_clf = Pipeline([("t", TfidfVectorizer(**tf)),
                                        ("c", LogisticRegression(**lr))])
            self._name_clf.fit(pos["text"], pos["label"])
        return self

    def predict(self, text: str) -> str:
        text = "" if text is None else str(text).strip()
        if not text or not self._has_clf:
            return ""
        proba = self._has_clf.predict_proba([text])[0]
        classes = list(self._has_clf.classes_)
        if 1 not in classes or proba[classes.index(1)] < self.prob_threshold:
            return ""
        if not self._name_clf:
            return ""
        return str(self._name_clf.predict([text])[0])


def fuzzy_match_fallback(text: str, candidates, score_cutoff: int = 82) -> str:
    if not _HAS_RAPIDFUZZ or not text or not text.strip() or not candidates:
        return ""
    match = process.extractOne(
        text, candidates, scorer=fuzz.partial_ratio, score_cutoff=score_cutoff
    )
    return match[0] if match else ""


# Lazily-trained predictors (trained once from train_labels.csv if present).
brand_predictor: NamePredictor | None = None
line_predictor: NamePredictor | None = None
_predictors_ready = False


def _ensure_predictors_trained() -> None:
    """Train brand_predictor / line_predictor once, from train_labels.csv if available."""
    global brand_predictor, line_predictor, _predictors_ready
    if _predictors_ready:
        return
    _predictors_ready = True

    from shared.data_utils import load_train_labels

    train_labels_df = load_train_labels()
    if train_labels_df is None:
        return

    tl_df = train_labels_df.copy()
    tl_df["ocr_text"] = tl_df["ocr_text"].astype(str).str.strip()
    tl_df["product_name"] = tl_df["product_name"].astype(str).str.strip()

    derived = tl_df.apply(_derive_brand_line, axis=1)
    tl_df = pd.concat([tl_df, derived], axis=1)

    brand_predictor = NamePredictor()
    brand_predictor.fit(tl_df["ocr_text"], tl_df["brand_derived"])

    line_predictor = NamePredictor(min_class_count=2)
    line_predictor.fit(tl_df["ocr_text"], tl_df["line_derived"])


def predict_brand_name(ocr_text: str) -> str:
    _ensure_predictors_trained()
    brand, _ = extract_brand_product(ocr_text)
    if not brand and brand_predictor is not None:
        brand = brand_predictor.predict(ocr_text)
    if not brand:
        brand = fuzzy_match_fallback(ocr_text, KNOWN_BRANDS)
    return normalize_brand(brand)


def predict_product_name(ocr_text: str, brand: str = "") -> str:
    if brand in BRAND_HAS_BUILTIN_LINE:
        return ""
    _ensure_predictors_trained()
    _, line = extract_brand_product(ocr_text)
    if not line and line_predictor is not None:
        line = line_predictor.predict(ocr_text)
    if not line and brand:
        line = fuzzy_match_fallback(ocr_text, KNOWN_LINES)
    return normalize_product(line)


# ---------------------------------------------------------------------------
# Discovery Layer (Cell 3b, tier 4) — heuristic brand/product mining for text
# that doesn't match any known rule, plus an in-session online cache.
# ---------------------------------------------------------------------------

_DISCOVERY_STOPWORDS = {
    "tin", "tức", "tin tức", "báo", "mới", "nóng", "khẩn", "sở", "y", "tế",
    "công", "ty", "cổ", "phần", "ctcp", "cty", "việt", "nam", "hà", "nội",
    "sài", "gòn", "thành", "phố", "quận", "huyện", "tỉnh", "review",
    "mukbang", "shorts", "video", "kênh", "channel", "cập", "nhật",
    "thông", "báo", "cảnh", "giác", "chú", "ý", "lưu", "người", "dân",
    "theo", "dõi", "chia", "sẻ", "group", "fanpage", "admin", "official",
    "tiktok", "facebook", "youtube", "instagram", "capcut", "content",
    "news", "breaking", "live", "trực", "tiếp", "hôm", "nay", "ngày",
    "tháng", "năm", "sáng", "chiều", "tối", "đêm", "giờ", "phút",
    "hút", "sốc", "kinh", "hoàng", "gây", "rúng", "động", "phẫn", "nộ",
    "nguy", "hiểm", "cấp", "khủng", "khiếp",
    "bất", "ngờ", "bùng", "nổ", "lan", "truyền", "viral", "hot",
    "singapore", "thái", "lan", "trung", "quốc", "nhật", "bản", "hàn",
    "mỹ", "pháp", "đức", "anh", "úc", "malaysia", "indonesia",
    "philippines", "lào", "campuchia", "châu", "á", "âu", "phi",
    "đà", "nẵng", "huế", "cần", "thơ", "hải", "phòng", "biên", "hòa",
    "trong", "ngoài", "trên", "dưới", "và", "hoặc", "nhưng", "là",
    "của", "cho", "với", "từ", "đến", "khi", "nếu", "vì", "do", "bởi",
    "này", "đó", "kia", "ấy", "đây", "đấy", "rồi", "đã", "sẽ", "đang",
    "không", "chưa", "còn", "cũng", "chỉ", "rất", "quá", "lắm",
    "tôi", "bạn", "chúng", "họ", "nó", "mình", "ta", "cụ", "ông", "bà",
    "anh", "chị", "em", "cô", "thu", "chi", "buông", "hàng", "đoàn",
    "kết", "cùng", "phải", "không", "cuộc", "đời", "tương", "lai",
    "vượt", "khó", "trao", "hi", "vọng", "tạo",
}

_DOMAIN_HINT_WORDS = {
    "sữa", "milk", "cà phê", "coffee", "trà", "tea", "pate", "patê",
    "đồ hộp", "xúc xích", "gia vị", "bánh kẹo", "nước giải khát",
    "thu hồi", "sản phẩm", "nhãn hàng", "thương hiệu", "dầu ăn",
    "thực phẩm", "tiệt trùng", "công thức", "dinh dưỡng", "hộp sữa",
}

_COMMERCIAL_CONTEXT_WORDS = {
    "ra mắt", "giảm giá", "khuyến mãi", "ưu đãi", "mua ngay",
    "chính thức có mặt", "phân phối", "nhập khẩu", "chính hãng",
    "lô hàng", "lỗi", "khủng hoảng truyền thông", "tẩy chay",
    "tăng giá", "mở bán", "ra đời", "bộ sưu tập", "dòng sản phẩm",
}

_TM_SYMBOLS = re.compile(r"[™®©]")
_VN_DIACRITIC_RE = re.compile(r"[\u00C0-\u024F\u1EA0-\u1EF9]")


def _tokenize_with_spans(text: str):
    return [(m.group(0), m.start(), m.end())
            for m in re.finditer(r"[A-Za-zÀ-ỹ0-9]+(?:[''\-][A-Za-zÀ-ỹ]+)*", text)]


def _looks_like_proper_noun(tok: str) -> bool:
    if not tok or tok.lower() in _DISCOVERY_STOPWORDS:
        return False
    if tok.isdigit():
        return False
    if len(tok) < 2:
        return False
    return tok[0].isupper()


def _extract_capitalized_runs(tokens_with_spans):
    runs, cur = [], []
    for tok, s, e in tokens_with_spans:
        if _looks_like_proper_noun(tok):
            cur.append((tok, s, e))
        else:
            if len(cur) >= 1:
                runs.append(cur)
            cur = []
    if len(cur) >= 1:
        runs.append(cur)

    out = []
    for r in runs:
        if not (1 <= len(r) <= 4):
            continue
        run_text = " ".join(t for t, _, _ in r)
        start_pos, end_pos = r[0][1], r[-1][2]
        out.append((run_text, start_pos, end_pos))
    return out


def _score_candidate(cand: str, text: str, start_pos: int) -> float:
    score = 0.0
    toks = cand.split()
    all_caps = all(t.isupper() for t in toks if len(t) > 1)

    window = text[max(0, start_pos - 30): start_pos + len(cand) + 30].lower()
    has_domain_hint = (any(hint in window for hint in _DOMAIN_HINT_WORDS)
                       or any(hint in window for hint in _COMMERCIAL_CONTEXT_WORDS))

    if all(t[0].isupper() for t in toks if t):
        score += 0.25

    tail = text[start_pos + len(cand): start_pos + len(cand) + 3]
    if _TM_SYMBOLS.search(tail):
        score += 0.35

    if start_pos <= 2:
        score += 0.10

    if has_domain_hint:
        score += 0.30

    if 2 <= len(cand) <= 25 and 1 <= len(toks) <= 3:
        score += 0.10
    else:
        score -= 0.15

    if all_caps and len(toks) >= 2 and not has_domain_hint:
        score -= 0.40

    if len(toks) == 1 and _VN_DIACRITIC_RE.search(cand) and not has_domain_hint:
        score -= 0.20

    return max(0.0, min(1.0, score))


def discover_brand_product(ocr_text: str, min_confidence: float = 0.45) -> tuple[str, str, float]:
    """Heuristically guess a (brand, product, confidence) not covered by any rule."""
    if not ocr_text or not ocr_text.strip():
        return "", "", 0.0

    tokens = _tokenize_with_spans(ocr_text)
    if not tokens:
        return "", "", 0.0

    runs = _extract_capitalized_runs(tokens)
    if not runs:
        return "", "", 0.0

    candidates = []
    for run_text, start_pos, end_pos in runs:
        sc = _score_candidate(run_text, ocr_text, start_pos)
        candidates.append((run_text, sc, start_pos))

    if not candidates:
        return "", "", 0.0

    candidates.sort(key=lambda x: (-x[1], x[2]))

    best_brand = candidates[0]

    window = ocr_text[max(0, best_brand[2] - 30): best_brand[2] + len(best_brand[0]) + 30].lower()
    has_strong_signal = (
        any(hint in window for hint in _DOMAIN_HINT_WORDS)
        or any(hint in window for hint in _COMMERCIAL_CONTEXT_WORDS)
        or bool(_TM_SYMBOLS.search(ocr_text[best_brand[2] + len(best_brand[0]): best_brand[2] + len(best_brand[0]) + 3]))
    )
    if not has_strong_signal:
        return "", "", best_brand[1]

    if best_brand[1] < min_confidence:
        return "", "", best_brand[1]

    brand_guess = best_brand[0]
    product_guess = ""
    for cand, sc, pos in candidates[1:]:
        if cand.lower() != brand_guess.lower() and sc >= min_confidence * 0.7:
            product_guess = cand
            break

    return brand_guess, product_guess, round(best_brand[1], 3)


_discovery_cache: dict = {}


def reset_discovery_cache() -> None:
    global _discovery_cache
    _discovery_cache = {}


def _cache_lookup(brand_guess: str):
    if not brand_guess:
        return None
    entry = _discovery_cache.get(brand_guess.strip().lower())
    return entry["canonical"] if entry else None


def _cache_learn(brand_guess: str, product_guess: str = "") -> None:
    if not brand_guess:
        return
    key = brand_guess.strip().lower()
    if key not in _discovery_cache:
        _discovery_cache[key] = {
            "canonical": brand_guess.strip(),
            "count": 0,
            "products": Counter(),
        }
    _discovery_cache[key]["count"] += 1
    if product_guess:
        _discovery_cache[key]["products"][product_guess.strip()] += 1


def _cache_search_in_text(ocr_text: str):
    if not ocr_text or not _discovery_cache:
        return None
    tl = ocr_text.lower()
    for key in sorted(_discovery_cache.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", tl):
            return _discovery_cache[key]["canonical"]
    return None


def discover_with_cache(ocr_text: str, min_confidence: float = 0.45) -> tuple[str, str, float]:
    """discover_brand_product() plus an in-session cache of previously
    discovered brands, so the same new brand is recognized consistently
    across images within one run (mirrors the notebook's online-cache demo).
    Note: the cache lives only in process memory and resets on restart."""
    brand_guess, product_guess, conf = discover_brand_product(ocr_text, min_confidence=0.0)

    if brand_guess:
        cached_canonical = _cache_lookup(brand_guess)
        if cached_canonical is not None:
            _cache_learn(brand_guess, product_guess)
            return cached_canonical, product_guess, max(conf, min_confidence)

        if conf >= min_confidence:
            _cache_learn(brand_guess, product_guess)
            return brand_guess, product_guess, conf

    direct_hit = _cache_search_in_text(ocr_text)
    if direct_hit is not None:
        _cache_learn(direct_hit)
        return direct_hit, "", min_confidence

    return "", "", conf


def print_discovery_cache_summary(min_count: int = 2) -> None:
    if not _discovery_cache:
        print("Cache is empty — no new brands learned yet.")
        return
    items = sorted(_discovery_cache.items(), key=lambda kv: -kv[1]["count"])
    print(f"Discovery cache: {len(items)} brands learned in this session.")
    shown = [it for it in items if it[1]["count"] >= min_count]
    print(f"  (showing {len(shown)} brands appearing >= {min_count} times)\n")
    for key, info in shown:
        top_products = ", ".join(p for p, _ in info["products"].most_common(3)) or "(none)"
        print(f"  '{info['canonical']}'  (x{info['count']})  top products: {top_products}")


def predict_brand_product(ocr_text: str) -> tuple[str, str]:
    """Full 3-tier prediction stack: regex rules -> TF-IDF -> fuzzy fallback.
    This is the version pipeline.py should call (supersedes the rules-only
    predict_brand_product() in brand_rules.py, exactly as in the notebook
    where Cell 3b's definition overrides Cell 3's).

    Note: discover_with_cache() (Discovery Layer, below) is an optional
    extra tier not used here — it is not part of the notebook pipeline.
    Call it explicitly if you want to opt in to heuristic brand/product
    mining for text that doesn't match any known rule."""
    brand = predict_brand_name(ocr_text)
    prod = predict_product_name(ocr_text, brand)
    return brand, prod


# ---------------------------------------------------------------------------
# VnRestorer — Vietnamese diacritic restoration (Cell 3c)
# ---------------------------------------------------------------------------

_VN_RE = _re.compile(r"[\u00C0-\u024F\u1EA0-\u1EF9]")


def _no_acc(s: str) -> str:
    s = _unorm2("NFD", s)
    s = _re.sub(r"[\u0300-\u036f]", "", s)
    return _unorm2("NFC", s.replace("đ", "d").replace("Đ", "D")).lower()


def _alpha_count(t: str) -> int:
    return len(_re.findall(r"[a-zA-Z\u00C0-\u024F\u1EA0-\u1EF9]", t))


def _vn_count(t: str) -> int:
    return len(_VN_RE.findall(t))


def _vn_ratio(t: str) -> float:
    if not t:
        return 0.0
    return _vn_count(t) / max(_alpha_count(t), 1)


_BASE_PHRASES = [
    ("doihoacitnhattu30namdoivoi", "đề nghị tù hoặc ít nhất tù 30 năm đối với"),
    ("itnhattu30nam",           "ít nhất tù 30 năm"),
    ("vidung130tanthitban",     "vì dùng 130 tấn thịt bẩn"),
    ("vuangan130tanthitlonbenh","vựa ngán 130 tấn thịt lợn bệnh"),
    ("130tanthitlonbenh",       "130 tấn thịt lợn bệnh"),
    ("thudoanhopthu",           "thủ đoạn hợp thức"),
    ("choagaysoc",              "choáng gây sốc"),
    ("tieuhuysanphamchebientuthitban", "tiêu hủy sản phẩm chế biến từ thịt bẩn"),
    ("sanphamchebientuthitban", "sản phẩm chế biến từ thịt bẩn"),
    ("chebientuthitban",        "chế biến từ thịt bẩn"),
    ("loanphaipatechebien",     "loạn phải pate chế biến"),
    ("patechebien",             "pate chế biến"),
    ("tuthitlonnhiembenh",      "từ thịt lợn nhiễm bệnh"),
    ("thitlonnhiembenh",        "thịt lợn nhiễm bệnh"),
    ("thitlonbenh",             "thịt lợn bệnh"),
    ("moinguytiemankhi",        "mối nguy tiềm ẩn khi"),
    ("bankhoekhong",            "bạn khỏe không"),
    ("suthatvethitthu",         "sự thật về thịt thối"),
    ("doilotdacsan",            "đội lốt đặc sản"),
    ("gocnhinnguoidan",         "góc nhìn người dân"),
    ("gocnhin",                 "góc nhìn"),
    ("nguoidan",                "người dân"),
    ("congdongmangdangquantamnhat", "cộng đồng mạng đang quan tâm nhất"),
    ("congdongmang",            "cộng đồng mạng"),
    ("dangquantam",              "đang quan tâm"),
    ("quantamnhat",              "quan tâm nhất"),
    ("noicjngdongmang",          "nơi cộng đồng mạng"),

    ("batkhancaptongiamdocvanhanviencongty", "bắt khẩn cấp tổng giám đốc và nhân viên công ty"),
    ("batkhancaptgdvanhanviencongty", "bắt khẩn cấp TGĐ và nhân viên công ty"),
    ("batkhancaptong",           "bắt khẩn cấp tổng"),
    ("batkhancap",                "bắt khẩn cấp"),
    ("tonggiamdoc",               "tổng giám đốc"),
    ("vanhanvien",                "và nhân viên"),
    ("phechuan",                  "phê chuẩn"),
    ("vksndtphaiphong",           "VKSND TP Hải Phòng"),

    ("congtycp",                  "Công ty CP"),

    ("banhmypatecotden",          "bánh mì pate cột đèn"),
    ("banhmypate",                "bánh mì pate"),
    ("patecotden",                "pate cột đèn"),
    ("patcotden",                 "pate cột đèn"),
    ("hatthuanchay",              "hạt thuần chay"),
    ("thuanchay",                 "thuần chay"),

    ("chauphinemphong4kho",       "châu Phi niêm phong 4 kho"),
    ("dichtachaup",               "dịch tả châu P"),

    ("highlandscoffeekhangdinhkhongsudung", "Highlands Coffee khẳng định không sử dụng"),
    ("khangdinhkhongsudung",      "khẳng định không sử dụng"),
    ("batkysanphamthit",          "bất kỳ sản phẩm thịt"),
    ("heochebiennaocuact",        "heo chế biến nào của CT"),
    ("highlandscoffeengungbantrasenvang", "Highlands Coffee ngừng bán trà sen vàng"),
    ("highlandscoffeengungban",   "Highlands Coffee ngừng bán"),
    ("ngungbantrasenvang",        "ngừng bán trà sen vàng"),
    ("trasenvang",                "trà sen vàng"),
    ("ngungban",                  "ngừng bán"),
    ("cungcap",                   "cung cấp"),
    ("codongthaimoi",             "có động thái mới"),
    ("nghivan",                   "nghi vấn"),

    ("uudaithanhto",              "ưu đãi thành tố"),
    ("banhngonmienphi",           "bánh ngon miễn phí"),
    ("mienphi",                   "miễn phí"),
    ("ketnoiniemtin",             "kết nối niềm tin"),
    ("bitancong",                 "bị tấn công"),
    ("lumcaptocgiamdoc",          "lùm xùm cấp tốc giám đốc"),
    ("dedontet",                  "để đón tết"),

    ("tam dung san xuat", "tạm dừng sản xuất"),
    ("san xuat",          "sản xuất"),
    ("san pham",          "sản phẩm"),
    ("thu hoi",           "thu hồi"),
    ("nguoi dung",        "người dùng"),
    ("nguoi tieu dung",   "người tiêu dùng"),
    ("khach hang",        "khách hàng"),
    ("cong ty",           "công ty"),
    ("co phan",           "cổ phần"),
    ("phat hien",         "phát hiện"),
    ("bi bat",            "bị bắt"),
    ("vu an",             "vụ án"),
    ("kiem tra",          "kiểm tra"),
    ("chat luong",        "chất lượng"),
    ("bao bi",            "bao bì"),
    ("nha san xuat",      "nhà sản xuất"),
    ("hang hoa",          "hàng hóa"),
    ("thi truong",        "thị trường"),
    ("an toan",           "an toàn"),
    ("suc khoe",          "sức khỏe"),
    ("dinh duong",        "dinh dưỡng"),
    ("tre em",            "trẻ em"),
    ("sua bot",           "sữa bột"),
    ("sua tuoi",          "sữa tươi"),
    ("thuc pham",         "thực phẩm"),
    ("nuoc mam",          "nước mắm"),
    ("banh mi",           "bánh mì"),
    ("ca phe",            "cà phê"),
    ("viet nam",          "Việt Nam"),
    ("ha noi",            "Hà Nội"),
    ("ho chi minh",       "Hồ Chí Minh"),
    ("da nang",           "Đà Nẵng"),
    ("ngay dang",         "ngày đăng"),
    ("nha hang",          "nhà hàng"),
    ("dia chi",           "địa chỉ"),
    ("gia re",            "giá rẻ"),
    ("chinh sach",        "chính sách"),
    ("doi voi",           "đối với"),
    ("hoac",              "hoặc"),
]

_BRAND_KEYS = ("canfoco", "canfood", "ha long", "halong", "nan ",
               "milo", "vinamilk", "aptamil", "highlands", "vissan",
               "do hop", "dohop", "hop ha", "hopha", "pate",
               "nestle", "dutch lady", "th true", "hipp", "friso")
_BRAND_STRINGS = ("halongcanfoco", "dohophalong", "congtydohophalong",
                   "vinamilk", "nestle", "aptamil", "highlands", "vissan",
                   "dutchlady", "thtrue", "hipp", "friso", "nan")


def _augment_from_train(labels_df) -> list[tuple[str, str]]:
    """Mine extra (no-accent-key -> accented phrase) pairs from train_labels,
    skipping anything that overlaps with brand names (those are handled by
    brand_rules.py, not the generic restorer)."""
    extra = []
    if labels_df is None:
        return extra
    seen = {k for k, _ in _BASE_PHRASES}
    for col in ("ocr_text", "product_name"):
        if col not in labels_df.columns:
            continue
        for raw in labels_df[col].astype(str).dropna():
            raw = raw.strip()
            if len(raw) < 6:
                continue
            if not _re.search(r"[\u00C0-\u024F\u1EA0-\u1EF9]", raw):
                continue
            words = raw.split()
            for n in (3, 2):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i:i + n])
                    if not _re.search(r"[\u00C0-\u024F\u1EA0-\u1EF9]", phrase):
                        continue
                    key = _no_acc(phrase)
                    key_compact = key.replace(" ", "")
                    if any(b in key for b in _BRAND_KEYS):
                        continue
                    if any(key_compact in bs or bs in key_compact for bs in _BRAND_STRINGS if len(key_compact) >= 4):
                        continue
                    if len(key) >= 6 and key not in seen:
                        seen.add(key)
                        extra.append((key, phrase))
    return extra


def _build_compiled_phrases() -> list[tuple["_re.Pattern", str]]:
    from shared.data_utils import load_train_labels

    phrases = list(_BASE_PHRASES) + _augment_from_train(load_train_labels())
    phrases.sort(key=lambda kv: len(kv[0]), reverse=True)

    compiled = []
    for src, tgt in phrases:
        if len(src) < 3:
            continue
        src_compact = src.replace(" ", "")
        if len(src_compact) >= 6:
            for pat in {src, src_compact}:
                compiled.append((_re.compile(_re.escape(pat), _re.IGNORECASE), tgt))
        else:
            regex = _re.compile(
                r"(?<![\w\u00C0-\u024F\u1EA0-\u1EF9])" + _re.escape(src) +
                r"(?![\w\u00C0-\u024F\u1EA0-\u1EF9])", _re.IGNORECASE)
            compiled.append((regex, tgt))
    return compiled


_COMPILED_PHRASES: list | None = None


def _get_compiled_phrases():
    global _COMPILED_PHRASES
    if _COMPILED_PHRASES is None:
        _COMPILED_PHRASES = _build_compiled_phrases()
    return _COMPILED_PHRASES


def restore_text(text: str) -> str:
    """Restore likely Vietnamese diacritics in OCR text that came back
    accent-stripped (e.g. 'sua tuoi' -> 'sữa tươi'). No-ops if the text
    already has enough Vietnamese diacritics (ratio >= 0.10)."""
    if not text:
        return text
    if _vn_ratio(text) >= 0.10:
        return text

    chars = list(text)
    protected = [False] * len(chars)

    def _apply(regex, tgt):
        nonlocal chars, protected
        result = "".join(chars)
        out_chars, out_protected = [], []
        pos = 0
        for m in regex.finditer(result):
            st, en = m.start(), m.end()
            if any(protected[st:en]):
                continue
            out_chars.append(result[pos:st])
            out_protected.extend(protected[pos:st])
            out_chars.append(tgt)
            out_protected.extend([True] * len(tgt))
            pos = en
        out_chars.append(result[pos:])
        out_protected.extend(protected[pos:])
        chars = list("".join(out_chars))
        protected = out_protected

    for regex, tgt in _get_compiled_phrases():
        _apply(regex, tgt)

    return "".join(chars)


if __name__ == "__main__":
    print("Brand predictor self-test (rules-only fallback, no train_labels):")
    t0 = time.perf_counter()
    for _ in range(200):
        predict_brand_product("Vinamilk Flex 180ml")
    print(f"Inference: {(time.perf_counter() - t0) * 1000 / 200:.3f} ms/img")

    print("\nDiscovery Layer self-test — brand/product NOT in rules:")
    for t in [
        "Singapore thu hồi lô sữa công thức của Nestle và Dumex vì nghi ngờ có độc tố",
        "Cocoxim Premium ra mắt dòng sản phẩm mới",
        "Heinz Ketchup giảm giá 20% toàn hệ thống",
    ]:
        b, p, conf = discover_brand_product(t)
        print(f"  '{t[:55]}' -> brand~'{b}' product~'{p}' (conf={conf})")

    print("\nVnRestorer self-test:")
    for txt in ["tam dung san xuat", "Nestlé NAN OPTIpro PLUS"]:
        print(f"  '{txt}' -> '{restore_text(txt)}'")