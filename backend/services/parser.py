import pdfplumber
from pdfplumber.utils.exceptions import PdfminerException
import defusedxml.ElementTree as DefusedET
from lxml import etree
from fastapi import UploadFile
from typing import List, Dict, Any, Optional
import io
import re

# Maximum file size in bytes (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Scene heading patterns (INT., EXT., INT/EXT., I/E., etc.)
SCENE_HEADING_RE = re.compile(
    r"^(INT\.|EXT\.|INT/EXT\.|I/E\.|INT\./EXT\.|EXT\./INT\.)"
    r".*",
    re.IGNORECASE,
)


def _classify_pdf_line(
    stripped: str,
    x0: Optional[float],
    x1: Optional[float],
    page_width: float,
    prev_type: Optional[str],
) -> str:
    """
    Classify a screenplay line using position and content heuristics.
    Matches FDX output types: heading, character, dialogue, parenthetical, action.
    """
    # Parenthetical: (something) - always detect first (overlaps with others)
    if stripped.startswith("(") and stripped.endswith(")") and len(stripped) <= 80:
        return "parenthetical"

    # Scene heading: INT./EXT. patterns
    if SCENE_HEADING_RE.match(stripped) and stripped.isupper():
        return "heading"

    # Character: centered + all caps + short (typically < 40 chars)
    if x0 is not None and x1 is not None and page_width > 0:
        line_center = (x0 + x1) / 2
        page_center = page_width / 2
        is_centered = abs(line_center - page_center) < page_width * 0.2
        if (
            is_centered
            and stripped.isupper()
            and len(stripped) <= 45
            and not stripped.startswith("(")
        ):
            return "character"

    # Context: dialogue follows character or parenthetical
    if prev_type in ("character", "parenthetical", "dialogue"):
        # Indented right of left margin = dialogue (dialogue has more indent than action)
        if x0 is not None and page_width > 0:
            # Left margin for action ~10% of page; dialogue ~15-20%
            if x0 > page_width * 0.12:
                return "dialogue"

    # Heuristic fallback when no position: indented + follows character
    if prev_type in ("character", "parenthetical", "dialogue"):
        return "dialogue"

    return "action"


async def parse_pdf(file: UploadFile) -> Dict[str, Any]:
    content = await file.read()

    # Check file size
    if len(content) > MAX_FILE_SIZE:
        raise ValueError(
            f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )

    # Check for empty file
    if len(content) == 0:
        raise ValueError("Empty file provided")

    lines = []
    characters = set()
    scenes = []

    try:
        pdf = pdfplumber.open(io.BytesIO(content))
    except (PdfminerException, Exception) as e:
        raise ValueError(f"Invalid or corrupted PDF file: {str(e)}")

    with pdf:
        for page in pdf.pages:
            page_width = float(page.width)

            # Try position-aware extraction first (same quality as FDX)
            try:
                text_lines = page.extract_text_lines(
                    strip=True,
                    return_chars=False,
                )
            except Exception:
                text_lines = []

            if text_lines:
                for line_obj in text_lines:
                    text = line_obj.get("text", "").strip()
                    if not text:
                        continue

                    x0 = line_obj.get("x0")
                    x1 = line_obj.get("x1")
                    prev_type = lines[-1]["type"] if lines else None

                    line_type = _classify_pdf_line(
                        text, x0, x1, page_width, prev_type
                    )

                    if line_type == "heading":
                        scenes.append({
                            "name": text,
                            "lineIndex": len(lines),
                        })
                    elif line_type == "character":
                        characters.add(text)

                    lines.append({
                        "type": line_type,
                        "content": text,
                        "original_text": text,
                    })
            else:
                # Fallback: layout text without positions
                text = page.extract_text(layout=True)
                if not text:
                    continue

                raw_lines = text.split("\n")
                prev_type = None

                for line in raw_lines:
                    stripped = line.strip()
                    if not stripped:
                        continue

                    line_type = _classify_pdf_line(
                        stripped, None, None, page_width, prev_type
                    )

                    if line_type == "heading":
                        scenes.append({
                            "name": stripped,
                            "lineIndex": len(lines),
                        })
                    elif line_type == "character":
                        characters.add(stripped)

                    lines.append({
                        "type": line_type,
                        "content": stripped,
                        "original_text": line,
                    })
                    prev_type = line_type

    return {
        "lines": lines,
        "characters": sorted(list(characters)),
        "scenes": scenes,
    }


async def parse_fdx(file: UploadFile) -> Dict[str, Any]:
    content = await file.read()

    # Check file size
    if len(content) > MAX_FILE_SIZE:
        raise ValueError(
            f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )

    # Check for empty file
    if len(content) == 0:
        raise ValueError("Empty file provided")

    # FDX is XML - use defusedxml for secure parsing (prevents XXE attacks)
    try:
        DefusedET.fromstring(content)
        root = etree.fromstring(content)
    except DefusedET.ParseError as e:
        raise ValueError(f"Invalid XML structure: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error parsing FDX file: {str(e)}")

    lines = []
    characters = set()
    scenes = []

    for paragraph in root.findall(".//Paragraph", namespaces=root.nsmap or {}):
        p_type = paragraph.get("Type", "Action")

        text_parts = []
        for text_node in paragraph.findall(".//Text", namespaces=root.nsmap or {}):
            if text_node.text:
                text_parts.append(text_node.text)

        full_text = "".join(text_parts).strip()
        if not full_text:
            continue

        internal_type = "action"
        if p_type == "Scene Heading":
            internal_type = "heading"
            scenes.append({"name": full_text, "lineIndex": len(lines)})
        elif p_type == "Character":
            internal_type = "character"
            characters.add(full_text)
        elif p_type == "Dialogue":
            internal_type = "dialogue"
        elif p_type == "Parenthetical":
            internal_type = "parenthetical"

        lines.append({
            "type": internal_type,
            "content": full_text,
            "original_text": full_text,
        })

    return {
        "lines": lines,
        "characters": sorted(list(characters)),
        "scenes": scenes,
    }
