#!/usr/bin/env python3
"""
db_pipeline.py - Database-driven psycholinguistic profiling pipeline.

Reads works, characters, and dialogues from slavodej.db, then generates
per-work folders with per-character profile reports (.md + .json).

Idempotent: skips characters whose .json profile already exists on disk.

Usage:
    python db_pipeline.py                       # uses default ../slavodej.db
    python db_pipeline.py /path/to/slavodej.db  # explicit database path
"""

import json
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from metrics import ensure_nltk, compute_metrics, CharacterMetrics
from profiler import (
    assign_profiles,
    build_profile_registry,
)
from pipeline import gemini_interpret

# Load env files for optional Gemini key
_env = Path(__file__).resolve().parent
load_dotenv(_env / ".env")
load_dotenv(_env.parent / "backend" / ".env")

# Default paths
DEFAULT_DB_PATH = _env.parent / "slavodej.db"
OUTPUT_ROOT = _env / "output"


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def sanitize_name(name: str) -> str:
    """Turn a work title or character name into a safe directory/file name."""
    safe = re.sub(r'[<>:"/\\|?*]', "", name)
    safe = safe.replace(" ", "_")
    safe = re.sub(r"_+", "_", safe).strip("_.")
    if not safe:
        safe = "unnamed"
    return safe


# ---------------------------------------------------------------------------
# Database queries
# ---------------------------------------------------------------------------

def fetch_works(conn: sqlite3.Connection) -> List[dict]:
    """Return all works that have at least one dialogue line."""
    cur = conn.execute("""
        SELECT w.*
        FROM works w
        WHERE EXISTS (SELECT 1 FROM dialogues d WHERE d.work_id = w.work_id)
        ORDER BY w.work_id
    """)
    cols = [desc[0] for desc in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_characters_for_work(
    conn: sqlite3.Connection, work_id: int
) -> List[dict]:
    """
    Return characters for a work that also have dialogue lines.
    Joins characters table with dialogues to skip characters without speech.
    """
    cur = conn.execute("""
        SELECT DISTINCT ch.*
        FROM characters ch
        INNER JOIN dialogues d
            ON d.work_id = ch.work_id AND d.character = ch.character
        WHERE ch.work_id = ?
        ORDER BY ch.character
    """, (work_id,))
    cols = [desc[0] for desc in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_dialogue_lines(
    conn: sqlite3.Connection, work_id: int, character: str
) -> List[str]:
    """Return all dialogue lines for a specific character in a work."""
    cur = conn.execute(
        "SELECT line FROM dialogues WHERE work_id = ? AND character = ?",
        (work_id, character),
    )
    return [row[0] for row in cur.fetchall() if row[0]]


# ---------------------------------------------------------------------------
# Per-character report generation
# ---------------------------------------------------------------------------

def generate_character_report(
    char_info: dict,
    work_info: dict,
    metrics: CharacterMetrics,
    matches: list,
    interpretation: Optional[str],
) -> str:
    """Build a Markdown profile report for a single character."""
    lines = []
    lines.append(f"# Character Profile: {char_info['character']}")
    lines.append(f"**Work:** {work_info['title']} ({work_info.get('year', 'N/A')})")
    if char_info.get("actor"):
        lines.append(f"**Actor:** {char_info['actor']}")
    if char_info.get("description"):
        lines.append(f"**Description:** {char_info['description']}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Profile assignments
    member_profiles = [m for m in matches if m.is_member]
    partial_profiles = [m for m in matches if m.is_partial]

    lines.append("## Assigned Profiles")
    lines.append("")
    if member_profiles:
        for m in member_profiles:
            lines.append(f"### {m.archetype.name} (score: {m.score:.3f})")
            lines.append(f"*{m.archetype.description}*")
            lines.append("")
            lines.append("Key feature contributions:")
            sorted_contrib = sorted(
                m.feature_contributions.items(), key=lambda x: x[1], reverse=True
            )
            for feat, val in sorted_contrib[:5]:
                lines.append(f"- {feat}: {val:.3f}")
            lines.append("")
    else:
        lines.append("No full archetype membership reached.")
        lines.append("")

    if partial_profiles:
        lines.append("### Partial Matches")
        for m in partial_profiles:
            lines.append(f"- {m.archetype.name} (score: {m.score:.3f})")
        lines.append("")

    # Metrics
    m = metrics
    lines.append("## Psycholinguistic Metrics")
    lines.append("")
    if m.warning:
        lines.append(f"> **Warning:** {m.warning}")
        lines.append("")
    lines.append(f"**Words:** {m.word_count} | **Sentences:** {m.sentence_count}")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Avg word length | {m.avg_word_length:.2f} |")
    lines.append(f"| Type-token ratio (TTR) | {m.type_token_ratio:.3f} |")
    lines.append(f"| Hapax ratio | {m.hapax_ratio:.3f} |")
    lines.append(f"| Avg sentence length | {m.avg_sentence_length:.1f} words |")
    lines.append(f"| Question ratio | {m.question_ratio:.2%} |")
    lines.append(f"| Exclamation ratio | {m.exclamation_ratio:.2%} |")
    lines.append(f"| Fragment ratio | {m.fragment_ratio:.2%} |")
    lines.append(f"| Sentiment (positive) | {m.sentiment_positive:.3f} |")
    lines.append(f"| Sentiment (negative) | {m.sentiment_negative:.3f} |")
    lines.append(f"| Sentiment (neutral) | {m.sentiment_neutral:.3f} |")
    lines.append(f"| Sentiment (compound) | {m.sentiment_compound:+.3f} |")
    lines.append(f"| Nouns % | {m.noun_pct:.1%} |")
    lines.append(f"| Verbs % | {m.verb_pct:.1%} |")
    lines.append(f"| Adjectives % | {m.adj_pct:.1%} |")
    lines.append(f"| Adverbs % | {m.adv_pct:.1%} |")
    lines.append("")

    # LIWC
    lines.append("### LIWC-like Categories")
    lines.append("| Category | Score |")
    lines.append("|----------|-------|")
    for cat in sorted(m.liwc.keys()):
        score = m.liwc[cat]
        bar = "#" * int(score * 50)
        lines.append(f"| {cat} | {score:.3f} {bar} |")
    lines.append("")

    # Top keywords
    if m.top_keywords:
        lines.append("### Top Keywords")
        lines.append(", ".join(
            f"`{kw}` ({count})" for kw, count in m.top_keywords[:10]
        ))
        lines.append("")

    # AI interpretation
    if interpretation:
        lines.append("## AI Interpretation")
        lines.append("")
        lines.append(interpretation)
        lines.append("")

    return "\n".join(lines)


def export_character_json(
    char_info: dict,
    work_info: dict,
    metrics: CharacterMetrics,
    matches: list,
) -> dict:
    """Build machine-readable JSON data for a single character."""
    m = metrics
    data = {
        "character": char_info["character"],
        "actor": char_info.get("actor"),
        "description": char_info.get("description"),
        "work": {
            "title": work_info["title"],
            "year": work_info.get("year"),
            "work_id": work_info["work_id"],
        },
        "generated": datetime.now().isoformat(),
        "word_count": m.word_count,
        "sentence_count": m.sentence_count,
        "warning": m.warning,
        "lexical": {
            "avg_word_length": round(m.avg_word_length, 3),
            "type_token_ratio": round(m.type_token_ratio, 3),
            "hapax_ratio": round(m.hapax_ratio, 3),
        },
        "syntactic": {
            "avg_sentence_length": round(m.avg_sentence_length, 2),
            "question_ratio": round(m.question_ratio, 3),
            "exclamation_ratio": round(m.exclamation_ratio, 3),
            "fragment_ratio": round(m.fragment_ratio, 3),
        },
        "sentiment": {
            "positive": round(m.sentiment_positive, 3),
            "negative": round(m.sentiment_negative, 3),
            "neutral": round(m.sentiment_neutral, 3),
            "compound": round(m.sentiment_compound, 3),
        },
        "pos_distribution": {
            "noun_pct": round(m.noun_pct, 3),
            "verb_pct": round(m.verb_pct, 3),
            "adj_pct": round(m.adj_pct, 3),
            "adv_pct": round(m.adv_pct, 3),
        },
        "liwc": {k: round(v, 4) for k, v in sorted(m.liwc.items())},
        "top_keywords": [{"word": w, "count": c} for w, c in m.top_keywords],
        "assigned_profiles": [],
    }

    for am in matches:
        if am.is_member or am.is_partial:
            data["assigned_profiles"].append({
                "profile": am.archetype.name,
                "score": am.score,
                "membership": "full" if am.is_member else "partial",
                "feature_contributions": am.feature_contributions,
            })

    return data


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_DB_PATH)

    if not os.path.exists(db_path):
        print(f"Error: Database not found: {db_path}")
        sys.exit(1)

    print("=" * 60)
    print("  DATABASE PROFILING PIPELINE")
    print("=" * 60)
    print(f"  DB: {db_path}")
    print(f"  Output: {OUTPUT_ROOT}")
    print()

    # Prepare NLTK
    print("[1/3] Preparing NLP resources...")
    ensure_nltk()

    # Connect to database
    conn = sqlite3.connect(db_path)

    works = fetch_works(conn)
    print(f"\n[2/3] Found {len(works)} works with dialogue.\n")

    total_profiled = 0
    total_skipped = 0

    for work in works:
        work_id = work["work_id"]
        title = work["title"]
        safe_title = sanitize_name(title)
        work_dir = OUTPUT_ROOT / safe_title

        characters = fetch_characters_for_work(conn, work_id)
        if not characters:
            continue

        print(f"--- {title} ({len(characters)} characters) ---")
        work_dir.mkdir(parents=True, exist_ok=True)

        # Save work metadata (idempotent)
        work_info_path = work_dir / "work_info.json"
        if not work_info_path.exists():
            with open(work_info_path, "w", encoding="utf-8") as f:
                json.dump(work, f, indent=2, ensure_ascii=False)

        for char_info in characters:
            char_name = char_info["character"]
            safe_char = sanitize_name(char_name)
            json_path = work_dir / f"{safe_char}.json"
            md_path = work_dir / f"{safe_char}.md"

            # Idempotency: skip if already profiled
            if json_path.exists():
                print(f"  SKIP {char_name} (already profiled)")
                total_skipped += 1
                continue

            dialogue_lines = fetch_dialogue_lines(conn, work_id, char_name)
            if not dialogue_lines:
                print(f"  SKIP {char_name} (no dialogue lines)")
                continue

            # Compute metrics
            metrics = compute_metrics(char_name, dialogue_lines)

            # Assign profiles
            assignments = assign_profiles({char_name: metrics})
            matches = assignments[char_name]

            member_names = [
                m.archetype.name for m in matches if m.is_member
            ]
            partial_names = [
                m.archetype.name for m in matches if m.is_partial
            ]

            # Optional Gemini interpretation per character
            registry = build_profile_registry(assignments)
            registry_text = "\n".join(
                f"### {e.profile_name}\n"
                f"Members: {e.members}\nPartial: {e.partial_members}"
                for e in registry
                if e.members or e.partial_members
            )
            metrics_summary = (
                f"### {char_name}\n"
                f"Words: {metrics.word_count}, "
                f"Sentiment compound: {metrics.sentiment_compound:+.3f}, "
                f"TTR: {metrics.type_token_ratio:.3f}, "
                f"Fragments: {metrics.fragment_ratio:.2%}, "
                f"LIWC anger: {metrics.liwc.get('anger', 0):.3f}, "
                f"LIWC cognitive: {metrics.liwc.get('cognitive', 0):.3f}, "
                f"LIWC power: {metrics.liwc.get('power', 0):.3f}, "
                f"LIWC risk: {metrics.liwc.get('risk_danger', 0):.3f}"
            )
            raw_dialogues = {char_name: metrics.raw_dialogue}
            interpretation = gemini_interpret(
                registry_text, metrics_summary, raw_dialogues
            )

            # Generate reports
            report_md = generate_character_report(
                char_info, work, metrics, matches, interpretation
            )
            char_json = export_character_json(char_info, work, metrics, matches)

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(report_md)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(char_json, f, indent=2, ensure_ascii=False)

            status = f"[{', '.join(member_names)}]" if member_names else "no full match"
            if partial_names and not member_names:
                status = f"partial: [{', '.join(partial_names)}]"
            print(f"  OK   {char_name} ({metrics.word_count} words) -> {status}")
            total_profiled += 1

    conn.close()

    print()
    print("=" * 60)
    print(f"  PIPELINE COMPLETE")
    print(f"  Profiled: {total_profiled} | Skipped: {total_skipped}")
    print(f"  Output:   {OUTPUT_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
