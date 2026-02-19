"""
extractor.py - Extract character dialogues from FDX and PDF screenplay files.

Produces a dict: { "CHARACTER_NAME": ["line1", "line2", ...], ... }
Each value is a list of raw dialogue strings for that character.
"""

import re
from pathlib import Path
from typing import Dict, List

from lxml import etree


# ---------------------------------------------------------------------------
# FDX (Final Draft XML) Extraction
# ---------------------------------------------------------------------------

def extract_from_fdx(filepath: str) -> Dict[str, List[str]]:
    """
    Parse an FDX file and return dialogue lines grouped by character.
    FDX paragraphs have explicit Type attributes (Character, Dialogue, etc.).
    """
    tree = etree.parse(filepath)
    root = tree.getroot()

    character_dialogues: Dict[str, List[str]] = {}
    current_character: str | None = None

    for paragraph in root.findall(".//Paragraph"):
        p_type = paragraph.get("Type", "Action")

        # Collect all <Text> children
        text_parts = []
        for text_node in paragraph.findall(".//Text"):
            if text_node.text:
                text_parts.append(text_node.text)
        full_text = "".join(text_parts).strip()
        if not full_text:
            continue

        if p_type == "Character":
            # Normalise: remove (CONT'D), (V.O.), (O.S.) etc.
            name = re.sub(r"\s*\(.*?\)\s*", "", full_text).strip().upper()
            if name:
                current_character = name
                character_dialogues.setdefault(current_character, [])

        elif p_type == "Dialogue" and current_character:
            # Skip placeholder / TODO lines
            if full_text.upper().startswith(("NEED DIALOGUE", "TODO")):
                continue
            character_dialogues[current_character].append(full_text)

        elif p_type in ("Scene Heading", "Action"):
            # Scene headings / action lines break the character context
            current_character = None

    return character_dialogues


# ---------------------------------------------------------------------------
# PDF Extraction  (uses pypdf for text, heuristics for structure)
# ---------------------------------------------------------------------------

_SCENE_HEADING_RE = re.compile(
    r"^(INT\.|EXT\.|INT/EXT\.|I/E\.|INT\./EXT\.|EXT\./INT\.)", re.IGNORECASE
)


def extract_from_pdf(filepath: str) -> Dict[str, List[str]]:
    """
    Parse a PDF screenplay and return dialogue lines grouped by character.
    Uses heuristic rules common to screenplay formatting.
    """
    from pypdf import PdfReader

    reader = PdfReader(filepath)
    all_lines: List[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_lines.extend(text.split("\n"))

    character_dialogues: Dict[str, List[str]] = {}
    current_character: str | None = None
    prev_type: str | None = None

    for raw_line in all_lines:
        stripped = raw_line.strip()
        if not stripped:
            continue

        # Classify line
        line_type = _classify_line(stripped, prev_type)

        if line_type == "character":
            name = re.sub(r"\s*\(.*?\)\s*", "", stripped).strip().upper()
            if name:
                current_character = name
                character_dialogues.setdefault(current_character, [])

        elif line_type == "dialogue" and current_character:
            if not stripped.upper().startswith(("NEED DIALOGUE", "TODO")):
                character_dialogues[current_character].append(stripped)

        elif line_type in ("heading", "action"):
            current_character = None

        prev_type = line_type

    return character_dialogues


def _classify_line(stripped: str, prev_type: str | None) -> str:
    """Simple heuristic screenplay line classifier."""
    if stripped.startswith("(") and stripped.endswith(")"):
        return "parenthetical"
    if _SCENE_HEADING_RE.match(stripped) and stripped == stripped.upper():
        return "heading"
    if (
        stripped == stripped.upper()
        and len(stripped) <= 45
        and not stripped.startswith("(")
        and not "." in stripped[:4]  # avoid matching action lines starting with abbrevs
    ):
        return "character"
    if prev_type in ("character", "parenthetical", "dialogue"):
        return "dialogue"
    return "action"


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def extract_dialogues(filepath: str) -> Dict[str, List[str]]:
    """
    Auto-detect file type and extract character dialogues.
    Returns { "CHARACTER_NAME": ["dialogue line 1", ...], ... }
    """
    p = Path(filepath)
    ext = p.suffix.lower()

    if ext == ".fdx":
        return extract_from_fdx(filepath)
    elif ext == ".pdf":
        return extract_from_pdf(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .fdx or .pdf")
