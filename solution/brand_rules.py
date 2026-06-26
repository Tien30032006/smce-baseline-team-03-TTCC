from __future__ import annotations

import re
from unicodedata import normalize as _unorm


def _strip_accent(s: str) -> str:
    s = _unorm("NFD", s)
    s = re.sub(r"[\u0300-\u036f]", "", s)
    return _unorm("NFC", s.replace("đ", "d").replace("Đ", "D")).lower()


BRAND_RULES = [
    (r"ha.?long.{0,25}canf[uo]c|canf[uo]c.{0,25}ha.?long|halongcanf",
     "Ha Long Canfoco"),
    (r"\bcanfoco\b|\bcanfood\b|canf[uo]co\b",
     "Ha Long Canfoco"),
    (r"(cong\s*ty|ctcp|co\s*phan|cty).{0,50}(do\s*hop|đồ\s*hộp).{0,30}ha.?long",
     "Đồ Hộp Hạ Long"),
    (r"(do\s*hop|đồ\s*hộp).{0,30}ha.?long",
     "Đồ Hộp Hạ Long"),
    (r"ha.?long.{0,30}(do\s*hop|đồ\s*hộp)",
     "Đồ Hộp Hạ Long"),
    (r"pate.{0,15}(c[oộ]t|cot).{0,10}(d[eèẻ]n|đèn)|patecot(?:den)?",
     "Pate Cột Đèn Hải Phòng"),
    (r"(c[oộ]t|cot).{0,10}(d[eèẻ]n|đèn).{0,20}(hai\s*phong|hải\s*phòng)",
     "Pate Cột Đèn Hải Phòng"),
    (r"ha.?long|halong",
     "Đồ Hộp Hạ Long"),

    (r"nan.{0,5}optipro.{0,5}plus|nan.{0,5}optiproplus",
     "Nestlé NAN OPTIpro PLUS"),
    (r"nan.{0,5}infinipro|nan.{0,5}infini\b",
     "Nestlé NAN INFINIPRO A2"),
    (r"nan.{0,5}supremepro|nan.{0,5}supreme",
     "Nestlé NAN SUPREMEpro"),
    (r"nan.{0,5}optipro\b",
     "Nestlé NAN OPTIpro"),
    (r"\bnan\b",
     "Nestlé NAN"),
    (r"\bbeba\b",
     "Nestlé BEBA"),
    (r"nestl[eé].{0,10}alfamino|\balfamino\b",
     "Nestlé Alfamino"),

    (r"\bmilo\b",                  "Nestlé Milo"),
    (r"nestle|nestlé",               "Nestlé"),
    (r"vinamilk",                    "Vinamilk"),
    (r"th\s*true|thtrue",           "TH True Milk"),
    (r"dutch\s*lady",                "Dutch Lady"),
    (r"\baptamil\b",               "Aptamil"),
    (r"similac",                     "Abbott Similac"),
    (r"\bensure\b",                "Abbott Ensure"),
    (r"pediasure",                   "Abbott PediaSure"),
    (r"glucerna",                    "Abbott Glucerna"),
    (r"\bfriso\b",                 "Friso"),
    (r"\bhipp\b",                  "HiPP"),
    (r"nutifood|\bnuti\b",         "Nutifood"),
    (r"optimum\s*gold",             "Optimum Gold"),
    (r"\bmeiji\b",                 "Meiji"),
    (r"\billuma\b",                "Illuma"),
    (r"\banlene\b",                "Anlene"),
    (r"\byomost\b",                "Yomost"),
    (r"\bfami\b",                  "Fami"),
    (r"lothamilk",                   "Lothamilk"),
    (r"ba\s*v[iì]\b|\bbavi\b",    "Ba Vì"),
    (r"dalat\s*milk",               "Đà Lạt Milk"),
    (r"\bkun\b",                   "Kun"),

    (r"highlands?.{0,5}coffee|\bhighlands\b", "Highlands Coffee"),
    (r"the\s*coffee\s*house|coffee\s*house", "Coffee House"),
    (r"phuc\s*long|phúc\s*long",              "Phúc Long"),

    (r"\bvissan\b",                "Vissan"),
    (r"chin[\s\-]*su|chinsu",       "Chin-Su"),
    (r"quang\s*h[oô]ng",            "Quang Hong Sardine"),
    (r"\bhafi\b",                  "Hafi"),
    (r"sardine|cá\s*mòi",           "Sardine"),
    (r"pate\s*minh\s*chay|minh\s*chay", "Pate Minh Chay"),

    (r"\bpate\b|\bpatê\b",
     "Pate Cột Đèn Hải Phòng"),
]

LINE_RULES = {
    "Ha Long Canfoco": [
        (r"pate.{0,10}(c[oộ]t|cot).{0,10}(d[eèẻ]n|đèn)|patecot", "Pate Cột Đèn"),
    ],
    "Vinamilk": [
        (r"\bflex\b",            "Flex"),
        (r"adm\s*gold",           "ADM Gold"),
        (r"\bsure\b",            "Sure"),
        (r"colosbaby",             "ColosBaby"),
        (r"dielac",                "Dielac"),
        (r"\bgrow\b",            "Grow"),
        (r"\bcanxi\b",           "Canxi"),
        (r"ong\s*tho|ông\s*thọ", "Ông Thọ"),
        (r"sua\s*chua|sữa\s*chua", "Sữa Chua"),
    ],
    "Nestlé Milo": [
        (r"3\s*in\s*1|3in1",     "3in1"),
    ],
    "TH True Milk": [
        (r"true\s*yogurt",        "True Yogurt"),
        (r"\bgrow\b",            "Grow"),
        (r"school\s*milk",        "School Milk"),
        (r"true\s*butter",        "True Butter"),
    ],
    "Dutch Lady": [
        (r"grow\s*\+?|grow\s*plus", "Grow+"),
        (r"complete",              "Complete"),
        (r"\bcanxi\b",           "Canxi"),
        (r"\bmom\b",             "Mum"),
    ],
    "Vissan": [
        (r"pate\s*heo",           "Pate Heo"),
        (r"pate\s*g[aà]\b",      "Pate Gà"),
        (r"xuc\s*xich|xúc\s*xích", "Xúc Xích"),
        (r"cá\s*x[oố]t\s*c[aà]", "Cá Xốt Cà"),
    ],
    "Highlands Coffee": [
        (r"tr[aà]\s*sen\s*v[aà]ng", "Trà Sen Vàng"),
        (r"tr[aà]\s*v[aả]i",         "Trà Vải"),
        (r"americano",                "Americano"),
        (r"phuc\s*kien|phúc\s*kiến","Phúc Kiến"),
    ],
    "Nutifood": [
        (r"growplus|grow\s*plus", "GrowPlus"),
        (r"\bpedia\b",           "Pedia"),
    ],
    "Aptamil": [
        (r"profutura",             "Profutura"),
    ],
    "Friso": [
        (r"\bgold\b",            "Gold"),
        (r"comfort",               "Comfort"),
        (r"prestige",              "Prestige"),
    ],
    "Abbott Ensure": [
        (r"\bgold\b",            "Gold"),
    ],
    "Anlene": [
        (r"\bgold\b",            "Gold"),
        (r"concentrate",           "Concentrate"),
    ],
    "Ba Vì": [
        (r"\bgold\b",            "Gold"),
    ],
    "Kun": [
        (r"chocolate",             "Chocolate"),
        (r"strawberry",            "Strawberry"),
    ],
}

# Brands whose canonical name already encodes the product line (e.g. NAN
# variants, Pate Cột Đèn, Đồ Hộp Hạ Long) — never look up a separate line.
BRAND_HAS_BUILTIN_LINE = {
    "Nestlé NAN OPTIpro PLUS", "Nestlé NAN INFINIPRO A2",
    "Nestlé NAN SUPREMEpro", "Nestlé NAN OPTIpro",
    "Pate Cột Đèn Hải Phòng", "Đồ Hộp Hạ Long",
}

NOISE_PRODUCTS = {
    "NS RECORDS", "Bình trữ sữa", "CapCut", "TikTok",
    "Finance", "ChatGPT", "LINH IT", "CP",
}

BRAND_NORMALIZE_MAP = [
    (r"^ha long canfood$",               "Ha Long Canfoco"),
    (r"^nan$|^sữa nan$",                 "Nestlé NAN"),
    (r"^nestle beba$",                   "Nestlé BEBA"),
]


def _find_brand(tl: str, tls: str) -> str:
    for pattern, brand in BRAND_RULES:
        if re.search(pattern, tl, re.IGNORECASE) or re.search(pattern, tls, re.IGNORECASE):
            return brand
    return ""


def _find_line(brand: str, tl: str, tls: str) -> str:
    for pattern, line in LINE_RULES.get(brand, []):
        if re.search(pattern, tl, re.IGNORECASE) or re.search(pattern, tls, re.IGNORECASE):
            return line
    return ""


def extract_brand_product(text: str) -> tuple[str, str]:
    """Return (brand, line) from raw OCR text using regex rules only."""
    if not text or not text.strip():
        return "", ""

    clean = re.sub(
        r"(công\s*ty|cty|ctcp|cổ\s*phần|tổng\s*giám\s*đốc|giám\s*đốc)\s*cp\b",
        "COTYCP", text, flags=re.IGNORECASE
    )
    tl = clean.lower()
    tls = _strip_accent(clean)

    brand = _find_brand(tl, tls)
    if not brand:
        return "", ""

    line = ""
    if brand not in BRAND_HAS_BUILTIN_LINE:
        line = _find_line(brand, tl, tls)

    return brand, line


def normalize_brand(name: str) -> str:
    if not name:
        return ""
    name = name.strip()
    for pat, canon in BRAND_NORMALIZE_MAP:
        if re.fullmatch(pat, name, re.IGNORECASE):
            return canon
    return name


def normalize_product(name: str) -> str:
    if not name:
        return ""
    name = name.strip()
    if name in NOISE_PRODUCTS:
        return ""
    name = re.sub(r"#\S+", "", name).strip()
    name = re.sub(r"\b\d+[.,]?\d*\s*(g|kg|ml|l|đ|%|k)\b", "", name, flags=re.IGNORECASE).strip()
    if len(name) > 60:
        return ""
    return name


def predict_brand_product(ocr_text: str) -> tuple[str, str]:
    """Rules-only brand/product prediction (no ML, no discovery layer)."""
    brand, line = extract_brand_product(ocr_text)
    return normalize_brand(brand), normalize_product(line)


# Convenience re-exports used by product_model.py / pipeline.py
KNOWN_BRANDS = sorted(set(b for _, b in BRAND_RULES))
KNOWN_LINES = sorted(set(l for rules in LINE_RULES.values() for _, l in rules))


if __name__ == "__main__":
    _tests = [
        ("HALONG CANFOCO PATE CỘT ĐÈN HẢI PHÒNG",     ("Ha Long Canfoco", "Pate Cột Đèn")),
        ("Công TY CP Đồ hộp Hạ Long bị bắt",          ("Đồ Hộp Hạ Long", "")),
        ("DO HOP HA LONG ISO 22000",                   ("Đồ Hộp Hạ Long", "")),
        ("PATE COT DEN CUA DO HOP HA LONG",            ("Đồ Hộp Hạ Long", "")),
        ("Sữa tươi tiệt trùng Vinamilk Flex 180ml",    ("Vinamilk", "Flex")),
        ("Vinamilk EST 1976 thông báo khẩn",           ("Vinamilk", "")),
        ("NESTLÉ MILO Chocolate Malt Drink 3in1",      ("Nestlé Milo", "3in1")),
        ("Vissan PATE HEO 170g combo 3 hộp",           ("Vissan", "Pate Heo")),
        ("Dutch Lady Grow+ 900g",                      ("Dutch Lady", "Grow+")),
        ("Ba Vì Gold 1L",                              ("Ba Vì", "Gold")),
        ("NAN OPTIPROPLUS thu hồi sữa",                ("Nestlé NAN OPTIpro PLUS", "")),
        ("LỜI NHẮN NHỦ VỀ NHÂN QUẢ VÀ DI SẢN",        ("", "")),
        ("LS",                                          ("", "")),
    ]
    passed = 0
    print("Brand/Product split rules self-test:")
    for text, exp in _tests:
        got = predict_brand_product(text)
        ok = got == exp
        passed += ok
        print(f"  {'OK' if ok else 'FAIL'} '{text[:48]}' -> {got}" + ("" if ok else f"  (exp: {exp})"))
    print(f"  {passed}/{len(_tests)} passed | Brand rules: {len(BRAND_RULES)} | Line rules: {sum(len(v) for v in LINE_RULES.values())}")