import re


def clean(text: str) -> str:
    """Ürün başlığını normalize eder: küçük harf, noktalama temizliği, boşluk düzeltme."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)  # model kodlarındaki '-' korunur
    text = re.sub(r"\s+", " ", text).strip()
    return text