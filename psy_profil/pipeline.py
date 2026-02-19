#!/usr/bin/env python3
"""
pipeline.py - Full psycholinguistic profiling pipeline for screenplay characters.

Usage:
    python pipeline.py <screenplay_file>       # FDX or PDF
    python pipeline.py ../Tets.fdx             # example

The pipeline:
  1. Extracts dialogue per character from the screenplay
  2. Computes extended psycholinguistic metrics per character
  3. Assigns characters to predefined profile archetypes (multi-membership)
  4. Computes data-driven similarity & clustering
  5. Optionally uses Gemini to interpret the profiles
  6. Generates a comprehensive Markdown report

Output:
  - profile_report_<timestamp>.md  (full human-readable report)
  - profile_data_<timestamp>.json  (machine-readable data)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

# Local modules
from extractor import extract_dialogues
from metrics import ensure_nltk, compute_metrics, CharacterMetrics, LIWC_CATEGORIES
from profiler import (
    assign_profiles,
    compute_similarity_matrix,
    cluster_characters,
    build_profile_registry,
    ARCHETYPES,
    MEMBERSHIP_THRESHOLD,
    PARTIAL_THRESHOLD,
)


# Load env
_env = Path(__file__).resolve().parent
load_dotenv(_env / ".env")
load_dotenv(_env.parent / "backend" / ".env")


TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================================
# GEMINI INTERPRETATION (optional)
# ============================================================================

def gemini_interpret(
    registry_text: str,
    metrics_summary: str,
    raw_dialogues: Dict[str, str],
) -> Optional[str]:
    """
    Use Gemini to provide a narrative interpretation of the profiles.
    Returns the interpretation text, or None if API is unavailable.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[INFO] GEMINI_API_KEY not set -- skipping AI interpretation.")
        return None

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("[INFO] google-genai not installed -- skipping AI interpretation.")
        return None

    system_prompt = """
YOU ARE A NARRATIVE PSYCHOLOGIST AND LITERARY PROFILER.

You are given:
1. A PROFILE REGISTRY showing predefined psychological archetypes and which screenplay
   characters matched each archetype, with quantitative similarity scores.
2. QUANTITATIVE METRICS per character (lexical, syntactic, sentiment, LIWC categories).
3. RAW DIALOGUE excerpts per character.

YOUR TASK:
- For each character, write a 3-5 sentence psychological interpretation that EXPLAINS
  why they matched their assigned profiles.
- CITE SPECIFIC EVIDENCE: reference exact metrics (e.g. "anger LIWC score of 0.12")
  and quote specific dialogue lines.
- EXPLAIN MULTI-MEMBERSHIP: if a character matches multiple profiles, explain what
  traits bridge those profiles.
- IDENTIFY DISCREPANCIES: if quantitative data contradicts the dialogue tone, note it.
- End with a brief ENSEMBLE ANALYSIS: how do these characters' profiles interact
  dramatically? What tensions or alliances do the profiles predict?

OUTPUT FORMAT: Clean Markdown with ## headers per character.
"""

    dialogue_excerpts = "\n\n".join(
        f"### {name}\n{text[:2000]}" for name, text in raw_dialogues.items()
    )

    user_payload = f"""
=== PROFILE REGISTRY ===
{registry_text}

=== QUANTITATIVE METRICS ===
{metrics_summary}

=== RAW DIALOGUE EXCERPTS ===
{dialogue_excerpts}
"""

    print("[AI] Sending data to Gemini for interpretation...")
    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.8,
        thinking_config=types.ThinkingConfig(thinking_budget=2048),
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_payload,
            config=config,
        )
        return response.text
    except Exception as e:
        print(f"[WARN] Gemini API error: {e}")
        return None


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(
    filepath: str,
    all_metrics: Dict[str, CharacterMetrics],
    assignments: Dict[str, list],
    sim_names: list,
    sim_matrix,
    clusters: Dict[int, list],
    interpretation: Optional[str],
) -> str:
    """Build the full Markdown report."""
    registry = build_profile_registry(assignments)

    lines = []
    lines.append(f"# Psycholinguistic Profile Report")
    lines.append(f"**Source:** `{filepath}`")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Characters analysed:** {len(all_metrics)}")
    lines.append("")

    # ---- Table of Contents ----
    lines.append("## Table of Contents")
    lines.append("1. [Methodology](#methodology)")
    lines.append("2. [Profile Registry](#profile-registry)")
    lines.append("3. [Character Metrics](#character-metrics)")
    lines.append("4. [Similarity Matrix](#similarity-matrix)")
    lines.append("5. [Data-Driven Clusters](#data-driven-clusters)")
    if interpretation:
        lines.append("6. [AI Interpretation](#ai-interpretation)")
    lines.append("")

    # ---- 1. Methodology ----
    lines.append("---")
    lines.append("## Methodology")
    lines.append("")
    lines.append("### Metrics Used")
    lines.append("| Family | Metrics | Source |")
    lines.append("|--------|---------|--------|")
    lines.append("| Lexical | avg word length, type-token ratio (TTR), hapax legomena ratio | NLTK tokenization |")
    lines.append("| Syntactic | avg sentence length, question %, exclamation %, fragment % | NLTK sent_tokenize |")
    lines.append("| Sentiment | VADER positive, negative, neutral, compound | NLTK VADER |")
    lines.append("| POS Distribution | noun %, verb %, adjective %, adverb % | NLTK POS tagger |")
    lines.append(f"| LIWC-like Categories | {len(LIWC_CATEGORIES)} categories (custom dictionaries) | Custom stem-matching |")
    lines.append("")
    lines.append("### LIWC-like Categories")
    for cat_name in sorted(LIWC_CATEGORIES.keys()):
        lines.append(f"- **{cat_name}**: stem-based matching against {len(LIWC_CATEGORIES[cat_name])} word stems")
    lines.append("")
    lines.append("### Profile Assignment Method")
    lines.append("Each predefined archetype has explicit criteria: a set of features with ideal values and weights.")
    lines.append("For each character-archetype pair:")
    lines.append("1. Compute `similarity = 1 - |actual_value - ideal_value|` per feature")
    lines.append("2. Multiply by the feature's weight")
    lines.append("3. Final score = `sum(similarity * weight) / sum(weights)`")
    lines.append(f"4. **Full membership** if score >= {MEMBERSHIP_THRESHOLD}")
    lines.append(f"5. **Partial match** if score >= {PARTIAL_THRESHOLD}")
    lines.append("6. A character CAN belong to **multiple** profiles simultaneously")
    lines.append("")
    lines.append("### Similarity & Clustering")
    lines.append("- Pairwise **cosine similarity** on normalised feature vectors")
    lines.append("- **Agglomerative clustering** (average linkage, cosine distance)")
    lines.append("")

    # ---- 2. Profile Registry ----
    lines.append("---")
    lines.append("## Profile Registry")
    lines.append("")
    lines.append("*This is the list of all profiles and their associated characters.*")
    lines.append("")

    for entry in registry:
        lines.append(f"### {entry.profile_name}")
        lines.append(f"*{entry.profile_description}*")
        lines.append("")

        if entry.members:
            lines.append("**Full Members:**")
            for char, score in entry.members:
                lines.append(f"- **{char}** (score: {score:.3f})")
        else:
            lines.append("**Full Members:** None")

        if entry.partial_members:
            lines.append("")
            lines.append("**Partial Matches:**")
            for char, score in entry.partial_members:
                lines.append(f"- {char} (score: {score:.3f})")

        lines.append("")
        lines.append("<details><summary>Criteria (click to expand)</summary>")
        lines.append("")
        lines.append("```")
        lines.append(entry.criteria_summary)
        lines.append("```")
        lines.append("</details>")
        lines.append("")

    # ---- 3. Character Metrics ----
    lines.append("---")
    lines.append("## Character Metrics")
    lines.append("")

    for name in sorted(all_metrics.keys()):
        m = all_metrics[name]
        lines.append(f"### {name}")
        if m.warning:
            lines.append(f"> **Warning:** {m.warning}")
            lines.append("")
        lines.append(f"**Words:** {m.word_count} | **Sentences:** {m.sentence_count}")
        lines.append("")

        # Profile assignments for this character
        char_matches = assignments.get(name, [])
        member_profiles = [am for am in char_matches if am.is_member]
        partial_profiles = [am for am in char_matches if am.is_partial]

        if member_profiles:
            lines.append("**Assigned Profiles:** " + ", ".join(
                f"{am.archetype.name} ({am.score:.3f})" for am in member_profiles
            ))
        if partial_profiles:
            lines.append("**Partial Matches:** " + ", ".join(
                f"{am.archetype.name} ({am.score:.3f})" for am in partial_profiles
            ))
        lines.append("")

        # Metrics table
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

        # LIWC categories
        lines.append("**LIWC-like Categories:**")
        lines.append("| Category | Score |")
        lines.append("|----------|-------|")
        for cat in sorted(m.liwc.keys()):
            score = m.liwc[cat]
            bar = "#" * int(score * 50)  # visual bar
            lines.append(f"| {cat} | {score:.3f} {bar} |")
        lines.append("")

        # Top keywords
        if m.top_keywords:
            lines.append("**Top Keywords:** " + ", ".join(
                f"`{kw}` ({count})" for kw, count in m.top_keywords[:10]
            ))
            lines.append("")

    # ---- 4. Similarity Matrix ----
    lines.append("---")
    lines.append("## Similarity Matrix")
    lines.append("")
    lines.append("Pairwise cosine similarity on normalised feature vectors:")
    lines.append("")

    # Header row
    header = "| |" + "|".join(f" **{n}** " for n in sim_names) + "|"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (len(sim_names) + 1)) + "|")

    for i, name in enumerate(sim_names):
        row = f"| **{name}** |"
        for j in range(len(sim_names)):
            val = sim_matrix[i][j]
            row += f" {val:.3f} |"
        lines.append(row)
    lines.append("")

    # ---- 5. Clusters ----
    lines.append("---")
    lines.append("## Data-Driven Clusters")
    lines.append("")
    lines.append("Characters grouped by feature similarity (agglomerative clustering):")
    lines.append("")

    for cluster_id in sorted(clusters.keys()):
        members = clusters[cluster_id]
        lines.append(f"- **Cluster {cluster_id + 1}:** {', '.join(members)}")
    lines.append("")

    # ---- 6. AI Interpretation ----
    if interpretation:
        lines.append("---")
        lines.append("## AI Interpretation")
        lines.append("")
        lines.append(interpretation)
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# JSON EXPORT
# ============================================================================

def export_json(
    filepath: str,
    all_metrics: Dict[str, CharacterMetrics],
    assignments: Dict[str, list],
    clusters: Dict[int, list],
) -> dict:
    """Build machine-readable JSON output."""
    registry = build_profile_registry(assignments)

    data = {
        "source_file": filepath,
        "generated": datetime.now().isoformat(),
        "methodology": {
            "metrics": [
                "lexical (TTR, hapax, avg_word_length)",
                "syntactic (avg_sentence_length, question_ratio, exclamation_ratio, fragment_ratio)",
                "sentiment (VADER pos/neg/neu/compound)",
                "POS distribution (noun/verb/adj/adv %)",
                f"LIWC-like ({len(LIWC_CATEGORIES)} categories)",
            ],
            "profile_assignment": "weighted_distance_scoring",
            "membership_threshold": MEMBERSHIP_THRESHOLD,
            "partial_threshold": PARTIAL_THRESHOLD,
            "similarity": "cosine_similarity",
            "clustering": "agglomerative_average_cosine",
        },
        "profiles": [],
        "characters": {},
        "clusters": {},
    }

    # Profiles
    for entry in registry:
        data["profiles"].append({
            "name": entry.profile_name,
            "description": entry.profile_description,
            "members": [{"character": c, "score": s} for c, s in entry.members],
            "partial_members": [{"character": c, "score": s} for c, s in entry.partial_members],
        })

    # Characters
    for name in sorted(all_metrics.keys()):
        m = all_metrics[name]
        char_data = {
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

        for am in assignments.get(name, []):
            if am.is_member or am.is_partial:
                char_data["assigned_profiles"].append({
                    "profile": am.archetype.name,
                    "score": am.score,
                    "membership": "full" if am.is_member else "partial",
                })

        data["characters"][name] = char_data

    # Clusters
    for cid, members in clusters.items():
        data["clusters"][f"cluster_{cid + 1}"] = members

    return data


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <screenplay.fdx|.pdf>")
        print("Example: python pipeline.py ../Tets.fdx")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    print("=" * 60)
    print("  PSYCHOLINGUISTIC PROFILING PIPELINE")
    print("=" * 60)

    # Step 0: Ensure NLTK resources
    print("\n[1/6] Preparing NLP resources...")
    ensure_nltk()

    # Step 1: Extract dialogues
    print(f"\n[2/6] Extracting character dialogues from: {filepath}")
    dialogues = extract_dialogues(filepath)

    if not dialogues:
        print("Error: No character dialogues found in the file.")
        sys.exit(1)

    for name, lines in sorted(dialogues.items()):
        word_count = sum(len(line.split()) for line in lines)
        print(f"  {name}: {len(lines)} lines, ~{word_count} words")

    # Step 2: Compute metrics
    print(f"\n[3/6] Computing psycholinguistic metrics...")
    all_metrics: Dict[str, CharacterMetrics] = {}
    for name, lines in dialogues.items():
        all_metrics[name] = compute_metrics(name, lines)
        m = all_metrics[name]
        status = f"OK ({m.word_count} words)"
        if m.warning:
            status = f"WARNING: {m.warning}"
        print(f"  {name}: {status}")

    # Step 3: Assign profiles
    print(f"\n[4/6] Assigning characters to profiles...")
    assignments = assign_profiles(all_metrics)

    for name in sorted(assignments.keys()):
        matches = assignments[name]
        member = [m for m in matches if m.is_member]
        partial = [m for m in matches if m.is_partial]
        if member:
            profiles = ", ".join(f"{m.archetype.name}({m.score:.3f})" for m in member)
            print(f"  {name} -> [{profiles}]")
        elif partial:
            profiles = ", ".join(f"{m.archetype.name}({m.score:.3f})" for m in partial)
            print(f"  {name} -> partial: [{profiles}]")
        else:
            print(f"  {name} -> no strong match")

    # Step 4: Similarity & Clustering
    print(f"\n[5/6] Computing similarity matrix & clusters...")
    sim_names, sim_matrix = compute_similarity_matrix(all_metrics)
    clusters = cluster_characters(all_metrics)

    for cid, members in sorted(clusters.items()):
        print(f"  Cluster {cid + 1}: {', '.join(members)}")

    # Step 5: Gemini interpretation (optional)
    print(f"\n[6/6] AI Interpretation...")
    registry = build_profile_registry(assignments)
    registry_text = "\n".join(
        f"### {e.profile_name}\nMembers: {e.members}\nPartial: {e.partial_members}"
        for e in registry
    )
    metrics_summary = "\n".join(
        f"### {name}\nWords: {m.word_count}, Sentiment compound: {m.sentiment_compound:+.3f}, "
        f"TTR: {m.type_token_ratio:.3f}, Fragments: {m.fragment_ratio:.2%}, "
        f"LIWC anger: {m.liwc.get('anger', 0):.3f}, LIWC cognitive: {m.liwc.get('cognitive', 0):.3f}, "
        f"LIWC power: {m.liwc.get('power', 0):.3f}, LIWC risk: {m.liwc.get('risk_danger', 0):.3f}"
        for name, m in sorted(all_metrics.items())
    )
    raw_dialogues = {name: m.raw_dialogue for name, m in all_metrics.items()}
    interpretation = gemini_interpret(registry_text, metrics_summary, raw_dialogues)

    # Step 6: Generate outputs
    print("\n--- Generating outputs ---")

    report_md = generate_report(
        filepath, all_metrics, assignments,
        sim_names, sim_matrix, clusters, interpretation,
    )
    report_file = f"profile_report_{TIMESTAMP}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"  Report: {report_file}")

    json_data = export_json(filepath, all_metrics, assignments, clusters)
    json_file = f"profile_data_{TIMESTAMP}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"  Data:   {json_file}")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
