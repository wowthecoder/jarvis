"""Answer normalization utilities for quasi-exact match scoring."""
import re
import unicodedata


def normalize_number(text: str) -> str:
    """Normalize numeric strings: remove commas, strip trailing zeros."""
    text = text.replace(",", "")
    try:
        f = float(text)
        # If it's a whole number, strip the decimal
        if f == int(f):
            return str(int(f))
        # Otherwise strip trailing zeros
        return str(f).rstrip("0").rstrip(".")
    except ValueError:
        return text


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for quasi-exact match comparison.

    Steps:
    - Unicode normalization
    - Strip leading/trailing whitespace
    - Lowercase
    - Remove leading articles (a, an, the)
    - Collapse multiple spaces
    - Normalize numbers
    """
    if not answer:
        return ""

    # Unicode normalization
    answer = unicodedata.normalize("NFKC", answer)
    answer = answer.strip().lower()

    # Remove leading articles
    answer = re.sub(r"^\s*(a|an|the)\s+", "", answer)

    # Collapse internal whitespace
    answer = re.sub(r"\s+", " ", answer).strip()

    # Try numeric normalization if it looks like a number
    if re.match(r"^-?[\d,]+\.?\d*$", answer.replace(" ", "")):
        answer = normalize_number(answer.replace(" ", ""))

    return answer
