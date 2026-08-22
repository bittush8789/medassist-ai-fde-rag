import re
import unicodedata
from typing import Optional


def clean_medical_text(text: str) -> str:
    """
    Cleans and normalizes raw text extracted from medical PDFs.
    
    1. Normalizes unicode characters (e.g., ligatures, Greek symbols).
    2. Strips leading/trailing whitespace.
    3. Replaces non-breaking spaces and irregular whitespace with standard spaces.
    4. Removes hyphenation artifacts across line breaks (e.g. 'hyper-\\ntension' -> 'hypertension').
    5. Collapses excessive consecutive newlines while preserving paragraph breaks.
    """
    if not text:
        return ""

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Replace zero-width spaces, non-breaking spaces, form feeds
    text = text.replace("\u200b", "").replace("\xa0", " ").replace("\x0c", "\n")

    # Fix broken hyphenated words at line endings (e.g. pharmaco-\nlogical -> pharmacological)
    text = re.sub(r'(\b\w+)-\n(\w+\b)', r'\1\2', text)

    # Replace multiple horizontal spaces/tabs with single space
    text = re.sub(r'[ \t]+', ' ', text)

    # Collapse 3 or more newlines into double newlines (preserving paragraphs)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Trim leading/trailing whitespace from each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines).strip()

    return text
