import re
import unicodedata


_LANG_TOKEN_RE = re.compile(r"<lang_([^>]+)>")


def normalize_language_code(lang: str) -> str:
    return lang.strip().lower().replace("-", "_")


def extract_language_token(text: str) -> str:
    match = _LANG_TOKEN_RE.search(text or "")
    if not match:
        return ""
    return normalize_language_code(match.group(1))


def normalize_text_for_eval(text: str, lang_hint: str = "") -> str:
    del lang_hint  # Reserved for future language-specific normalization rules.
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    kept: list[str] = []
    for ch in normalized:
        if ch.isspace():
            kept.append(" ")
            continue
        category = unicodedata.category(ch)
        if category[:1] in {"L", "M", "N"}:
            kept.append(ch)
            continue
        kept.append(" ")
    return " ".join("".join(kept).split())
