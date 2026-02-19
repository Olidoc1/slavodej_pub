"""
profiler.py - Profile definitions, similarity computation, character assignment.

This module answers the core design questions:
  - How are profiles defined?
  - How can a character belong to multiple profiles?
  - What determines similarity between characters?
  - What are the explicit criteria / weights?

Two complementary approaches:
  1. ARCHETYPE MATCHING  - predefined profile archetypes with prototype feature
     patterns.  Each character is scored against each archetype via weighted
     cosine similarity.  If the score exceeds a threshold, the character
     belongs to that profile.  A character CAN match multiple archetypes.

  2. DATA-DRIVEN CLUSTERING - agglomerative clustering on the feature vectors
     discovers natural groupings.  This shows which characters are most similar
     to each other regardless of predefined archetypes.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

from metrics import CharacterMetrics, LIWC_CATEGORIES


# ============================================================================
# 1. ARCHETYPE DEFINITIONS
# ============================================================================
# Each archetype has:
#   - name: human-readable label
#   - description: what this profile means psychologically
#   - prototype: a partial feature vector (only the features that define it)
#   - weights: how much each feature matters for this archetype (0..1)
#
# Features not listed in the prototype get weight=0 (don't affect the match).
# This makes the criteria EXPLICIT and TRANSPARENT.
# ============================================================================

@dataclass
class ArchetypeDefinition:
    name: str
    description: str
    # key -> (ideal_value, weight)
    # ideal_value in [0, 1], weight in [0, 1]
    criteria: Dict[str, Tuple[float, float]]

    def describe_criteria(self) -> str:
        """Human-readable description of what drives this archetype."""
        lines = []
        sorted_criteria = sorted(
            self.criteria.items(), key=lambda x: x[1][1], reverse=True
        )
        for feat, (ideal, weight) in sorted_criteria:
            direction = "high" if ideal > 0.5 else "low"
            lines.append(
                f"  - {feat}: {direction} (ideal={ideal:.2f}, weight={weight:.2f})"
            )
        return "\n".join(lines)


# The predefined archetypes
ARCHETYPES: List[ArchetypeDefinition] = [
    ArchetypeDefinition(
        name="The Aggressor",
        description=(
            "Dominant, confrontational personality. Uses short, direct sentences. "
            "High negative emotion and anger vocabulary. Speaks to control."
        ),
        criteria={
            "fragment_ratio": (0.8, 0.9),        # short clipped sentences
            "avg_sentence_length": (0.2, 0.7),    # low = short sentences
            "sentiment_negative": (0.8, 0.8),     # high negative affect
            "liwc_anger": (0.8, 1.0),             # anger words
            "liwc_power": (0.7, 0.8),             # power/dominance words
            "liwc_negative_emotion": (0.8, 0.7),  # negative emotion
            "exclamation_ratio": (0.7, 0.5),      # commands, exclamations
            "question_ratio": (0.2, 0.3),         # doesn't ask, tells
        },
    ),
    ArchetypeDefinition(
        name="The Intellectual",
        description=(
            "Analytical, deliberate thinker. Uses longer, complex sentences. "
            "High cognitive vocabulary. Tends toward neutral or measured emotion."
        ),
        criteria={
            "avg_sentence_length": (0.8, 0.9),    # long sentences
            "liwc_cognitive": (0.8, 1.0),          # cognitive process words
            "type_token_ratio": (0.8, 0.7),        # rich vocabulary
            "adj_pct": (0.7, 0.6),                 # descriptive
            "sentiment_neutral": (0.7, 0.5),       # measured emotion
            "fragment_ratio": (0.2, 0.6),          # avoids fragments
            "liwc_certainty": (0.7, 0.5),          # speaks with authority
        },
    ),
    ArchetypeDefinition(
        name="The Emotionalist",
        description=(
            "Driven by feelings. High affective language, both positive and negative. "
            "Emotional volatility visible in sentiment swings. Expressive syntax."
        ),
        criteria={
            "liwc_positive_emotion": (0.7, 0.8),   # positive emotion words
            "liwc_negative_emotion": (0.6, 0.6),    # also negative emotion
            "liwc_sadness": (0.6, 0.7),             # sadness vocabulary
            "sentiment_positive": (0.6, 0.6),       # positive sentiment
            "exclamation_ratio": (0.7, 0.7),        # expressive
            "liwc_social": (0.7, 0.6),              # social/relational
            "question_ratio": (0.5, 0.4),           # asks questions
        },
    ),
    ArchetypeDefinition(
        name="The Controller",
        description=(
            "Strategic, authoritative speaker. Uses power and achievement language. "
            "Measured emotion, factual delivery. Commands rather than requests."
        ),
        criteria={
            "liwc_power": (0.8, 1.0),              # power vocabulary
            "liwc_achievement": (0.7, 0.8),         # achievement-oriented
            "liwc_certainty": (0.8, 0.7),           # speaks with certainty
            "sentiment_neutral": (0.7, 0.6),        # emotionally flat
            "liwc_cognitive": (0.6, 0.5),           # some calculation
            "question_ratio": (0.3, 0.5),           # tells, doesn't ask
            "fragment_ratio": (0.5, 0.4),           # mixed sentence length
        },
    ),
    ArchetypeDefinition(
        name="The Survivor",
        description=(
            "Alert to danger, risk-aware. Uses anxiety and risk/danger vocabulary. "
            "Pragmatic speech, short sentences, vigilant mindset."
        ),
        criteria={
            "liwc_anxiety": (0.7, 0.9),            # anxiety words
            "liwc_risk_danger": (0.8, 1.0),         # danger/risk vocabulary
            "liwc_negative_emotion": (0.6, 0.6),    # negative affect
            "fragment_ratio": (0.6, 0.6),           # terse
            "sentiment_negative": (0.6, 0.5),       # negative sentiment
            "question_ratio": (0.5, 0.4),           # checking, verifying
            "liwc_tentativeness": (0.6, 0.5),       # hedging
        },
    ),
    ArchetypeDefinition(
        name="The Pragmatist",
        description=(
            "Factual, no-nonsense speaker. Neutral sentiment, achievement-oriented. "
            "Concrete vocabulary, moderate sentence length, low emotional affect."
        ),
        criteria={
            "sentiment_neutral": (0.8, 0.9),        # emotionally flat
            "sentiment_compound_shifted": (0.5, 0.7), # balanced compound
            "liwc_achievement": (0.6, 0.6),          # results-oriented
            "noun_pct": (0.7, 0.5),                  # concrete nouns
            "liwc_positive_emotion": (0.2, 0.5),     # low positive emotion
            "liwc_negative_emotion": (0.2, 0.5),     # low negative emotion
            "type_token_ratio": (0.5, 0.3),          # average vocabulary
        },
    ),
    ArchetypeDefinition(
        name="The Manipulator",
        description=(
            "Deceptive, calculating speaker. Uses social and cognitive language to "
            "influence others. Tentative phrasing masks true intent. Asks questions "
            "to probe and control the conversation indirectly."
        ),
        criteria={
            "liwc_social": (0.8, 1.0),              # social engineering
            "liwc_cognitive": (0.7, 0.9),            # calculating mind
            "liwc_tentativeness": (0.7, 0.8),        # hedging, plausible deniability
            "question_ratio": (0.7, 0.8),            # probing questions
            "liwc_positive_emotion": (0.6, 0.6),     # surface charm
            "sentiment_compound_shifted": (0.6, 0.5), # slightly positive facade
            "liwc_certainty": (0.2, 0.6),            # avoids definitive statements
            "exclamation_ratio": (0.2, 0.4),         # rarely raises voice
        },
    ),
    ArchetypeDefinition(
        name="The Protector",
        description=(
            "Selfless, protective personality. High social and positive emotion language. "
            "Achievement-oriented in service of others. Speaks with warmth and urgency "
            "when those they care about are threatened."
        ),
        criteria={
            "liwc_social": (0.8, 1.0),              # deeply social/relational
            "liwc_positive_emotion": (0.7, 0.8),     # warmth, care
            "liwc_risk_danger": (0.6, 0.7),          # aware of threats to others
            "liwc_achievement": (0.6, 0.6),          # driven to help/succeed
            "exclamation_ratio": (0.6, 0.6),         # urgent when needed
            "sentiment_positive": (0.6, 0.5),        # generally positive
            "liwc_power": (0.5, 0.4),                # protective authority
            "liwc_anger": (0.2, 0.4),                # low personal anger
        },
    ),
    ArchetypeDefinition(
        name="The Rebel",
        description=(
            "Defiant outsider who rejects authority and convention. Uses negative emotion "
            "and risk language. Low social integration. Short, punchy sentences that "
            "challenge the status quo."
        ),
        criteria={
            "liwc_negative_emotion": (0.7, 0.9),    # dissatisfaction
            "liwc_anger": (0.6, 0.8),                # defiance
            "liwc_risk_danger": (0.6, 0.7),          # comfort with danger
            "liwc_social": (0.2, 0.7),               # low social conformity
            "liwc_certainty": (0.7, 0.6),            # strong convictions
            "fragment_ratio": (0.7, 0.6),            # blunt, clipped speech
            "liwc_power": (0.3, 0.5),                # rejects power structures
            "liwc_tentativeness": (0.2, 0.5),        # no hedging
        },
    ),
    ArchetypeDefinition(
        name="The Victim",
        description=(
            "Passive, suffering personality. High sadness and anxiety vocabulary. "
            "Tentative speech patterns, negative sentiment. Speaks from a position "
            "of helplessness and resignation."
        ),
        criteria={
            "liwc_sadness": (0.8, 1.0),             # deep sadness
            "liwc_anxiety": (0.7, 0.9),              # chronic worry
            "liwc_negative_emotion": (0.8, 0.8),     # pervasive negativity
            "liwc_tentativeness": (0.7, 0.7),        # uncertain, hesitant
            "sentiment_negative": (0.7, 0.7),        # negative sentiment
            "liwc_power": (0.1, 0.6),                # powerless
            "liwc_achievement": (0.1, 0.5),          # no sense of accomplishment
            "exclamation_ratio": (0.2, 0.4),         # subdued expression
        },
    ),
    ArchetypeDefinition(
        name="The Philosopher",
        description=(
            "Reflective, questioning mind. High cognitive and tentative language, "
            "long sentences exploring ideas. Asks questions not to probe others "
            "but to probe existence. Measured, contemplative tone."
        ),
        criteria={
            "liwc_cognitive": (0.9, 1.0),            # deep thinking
            "liwc_tentativeness": (0.7, 0.8),        # exploring, not asserting
            "avg_sentence_length": (0.8, 0.9),       # long, flowing thoughts
            "question_ratio": (0.6, 0.7),            # existential questioning
            "type_token_ratio": (0.8, 0.7),          # rich vocabulary
            "fragment_ratio": (0.1, 0.6),            # avoids fragments
            "sentiment_neutral": (0.7, 0.5),         # emotionally detached
            "adj_pct": (0.7, 0.5),                   # descriptive language
        },
    ),
    ArchetypeDefinition(
        name="The Cynic",
        description=(
            "Disillusioned, sardonic speaker. Mixes cognitive and negative emotion "
            "language. Uses irony -- positive words in negative contexts. Questions "
            "motives and sincerity. Moderate vocabulary complexity."
        ),
        criteria={
            "liwc_negative_emotion": (0.7, 0.9),    # underlying negativity
            "liwc_cognitive": (0.6, 0.8),            # analytical dismissal
            "sentiment_negative": (0.6, 0.7),        # negative tone
            "question_ratio": (0.5, 0.6),            # rhetorical questions
            "liwc_positive_emotion": (0.3, 0.5),     # ironic positivity
            "liwc_social": (0.3, 0.5),               # socially withdrawn
            "liwc_certainty": (0.6, 0.5),            # certain about the worst
            "type_token_ratio": (0.7, 0.4),          # somewhat varied vocabulary
        },
    ),
    ArchetypeDefinition(
        name="The Zealot",
        description=(
            "Fanatically convinced speaker. Extreme certainty, power language, "
            "high exclamation ratio. Speaks in absolutes with no room for doubt. "
            "Achievement-driven rhetoric, often at the expense of nuance."
        ),
        criteria={
            "liwc_certainty": (0.9, 1.0),           # absolute conviction
            "exclamation_ratio": (0.8, 0.9),         # fervent delivery
            "liwc_power": (0.7, 0.8),                # authority, dominance
            "liwc_achievement": (0.7, 0.7),          # mission-driven
            "liwc_tentativeness": (0.1, 0.7),        # zero hedging
            "question_ratio": (0.1, 0.6),            # declares, never asks
            "sentiment_compound_shifted": (0.7, 0.5), # emotionally charged positive
            "fragment_ratio": (0.6, 0.4),            # punchy declarations
        },
    ),
    ArchetypeDefinition(
        name="The Mentor",
        description=(
            "Wise guide figure. Combines cognitive depth with social warmth. "
            "Uses achievement language to encourage, certainty to reassure. "
            "Longer sentences that teach and explain."
        ),
        criteria={
            "liwc_cognitive": (0.7, 1.0),            # wisdom, understanding
            "liwc_social": (0.7, 0.9),               # connected to others
            "liwc_achievement": (0.7, 0.7),          # encouraging accomplishment
            "liwc_positive_emotion": (0.6, 0.7),     # supportive warmth
            "avg_sentence_length": (0.7, 0.7),       # explanatory speech
            "liwc_certainty": (0.6, 0.6),            # confident guidance
            "question_ratio": (0.4, 0.5),            # Socratic questions
            "liwc_anger": (0.1, 0.5),                # patient, not angry
        },
    ),
    ArchetypeDefinition(
        name="The Peacemaker",
        description=(
            "Conciliatory, harmony-seeking speaker. High positive emotion and social "
            "language. Tentative phrasing to avoid conflict. Low anger, low power "
            "assertion. Asks questions to understand, not to challenge."
        ),
        criteria={
            "liwc_positive_emotion": (0.8, 1.0),    # positivity
            "liwc_social": (0.8, 0.9),               # relational harmony
            "liwc_anger": (0.1, 0.8),                # actively avoids anger
            "liwc_tentativeness": (0.6, 0.7),        # diplomatic hedging
            "liwc_power": (0.1, 0.7),                # non-dominant
            "question_ratio": (0.5, 0.6),            # inclusive questioning
            "sentiment_positive": (0.7, 0.6),        # warm sentiment
            "liwc_negative_emotion": (0.1, 0.5),     # suppresses negativity
        },
    ),
    ArchetypeDefinition(
        name="The Narcissist",
        description=(
            "Self-absorbed, grandiose speaker. High power and achievement language "
            "centred on self. Certainty in own superiority. Low social engagement "
            "except as audience. Positive self-sentiment masking contempt for others."
        ),
        criteria={
            "liwc_power": (0.8, 1.0),               # personal dominance
            "liwc_achievement": (0.8, 0.9),          # self-aggrandisement
            "liwc_certainty": (0.8, 0.8),            # unshakeable self-belief
            "liwc_social": (0.2, 0.7),               # low genuine connection
            "sentiment_compound_shifted": (0.7, 0.6), # positive self-view
            "liwc_positive_emotion": (0.5, 0.5),     # selective positivity
            "liwc_anxiety": (0.1, 0.5),              # no vulnerability
            "liwc_sadness": (0.1, 0.5),              # no vulnerability
        },
    ),
]


# ============================================================================
# 2. SCORING: CHARACTER vs ARCHETYPE
# ============================================================================

# Threshold: character belongs to archetype if score >= this value
MEMBERSHIP_THRESHOLD = 0.45

# Minimum score to be listed as "partial match"
PARTIAL_THRESHOLD = 0.30


@dataclass
class ArchetypeMatch:
    archetype: ArchetypeDefinition
    score: float          # 0..1 weighted similarity
    is_member: bool       # score >= MEMBERSHIP_THRESHOLD
    is_partial: bool      # PARTIAL_THRESHOLD <= score < MEMBERSHIP_THRESHOLD
    feature_contributions: Dict[str, float] = field(default_factory=dict)


def score_character_against_archetype(
    metrics: CharacterMetrics,
    archetype: ArchetypeDefinition,
) -> ArchetypeMatch:
    """
    Score how well a character matches an archetype.

    Method: Weighted distance scoring.
    For each criterion in the archetype:
      - Get the character's actual value for that feature
      - Compute similarity = 1 - |actual - ideal|
      - Multiply by the criterion's weight
    Final score = weighted_sum / sum_of_weights

    This makes the process fully QUANTITATIVE and TRANSPARENT.
    """
    feature_names = metrics.feature_vector_names()
    feature_values = metrics.feature_vector()
    feat_map = dict(zip(feature_names, feature_values))

    weighted_sum = 0.0
    total_weight = 0.0
    contributions: Dict[str, float] = {}

    for feat_name, (ideal, weight) in archetype.criteria.items():
        actual = feat_map.get(feat_name, 0.0)
        similarity = 1.0 - abs(actual - ideal)
        contribution = similarity * weight
        weighted_sum += contribution
        total_weight += weight
        contributions[feat_name] = round(contribution / weight if weight else 0, 3)

    score = weighted_sum / total_weight if total_weight > 0 else 0.0

    return ArchetypeMatch(
        archetype=archetype,
        score=round(score, 4),
        is_member=score >= MEMBERSHIP_THRESHOLD,
        is_partial=PARTIAL_THRESHOLD <= score < MEMBERSHIP_THRESHOLD,
        feature_contributions=contributions,
    )


def assign_profiles(
    all_metrics: Dict[str, CharacterMetrics],
) -> Dict[str, List[ArchetypeMatch]]:
    """
    For each character, score against all archetypes.
    Returns { "CHARACTER": [ArchetypeMatch, ...] } sorted by score desc.
    """
    results: Dict[str, List[ArchetypeMatch]] = {}

    for char_name, metrics in all_metrics.items():
        matches = []
        for archetype in ARCHETYPES:
            match = score_character_against_archetype(metrics, archetype)
            matches.append(match)
        matches.sort(key=lambda m: m.score, reverse=True)
        results[char_name] = matches

    return results


# ============================================================================
# 3. DATA-DRIVEN SIMILARITY & CLUSTERING
# ============================================================================

def compute_similarity_matrix(
    all_metrics: Dict[str, CharacterMetrics],
) -> Tuple[List[str], np.ndarray]:
    """
    Compute pairwise cosine similarity between all characters.
    Returns (character_names, similarity_matrix).
    """
    names = sorted(all_metrics.keys())
    vectors = []
    for name in names:
        vectors.append(all_metrics[name].feature_vector())

    X = np.array(vectors)

    # Normalise features to [0, 1] across characters
    if X.shape[0] > 1:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    sim_matrix = cosine_similarity(X)
    return names, sim_matrix


def cluster_characters(
    all_metrics: Dict[str, CharacterMetrics],
    n_clusters: Optional[int] = None,
    distance_threshold: float = 0.5,
) -> Dict[int, List[str]]:
    """
    Agglomerative clustering on feature vectors.
    If n_clusters is None, uses distance_threshold to auto-determine.
    Returns { cluster_id: [character_names] }.
    """
    names = sorted(all_metrics.keys())

    if len(names) < 2:
        return {0: names}

    vectors = []
    for name in names:
        vectors.append(all_metrics[name].feature_vector())

    X = np.array(vectors)

    if X.shape[0] > 1:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    if n_clusters is not None:
        n_clusters = min(n_clusters, len(names))
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric="cosine", linkage="average"
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )

    labels = clustering.fit_predict(X)

    clusters: Dict[int, List[str]] = {}
    for name, label in zip(names, labels):
        clusters.setdefault(int(label), []).append(name)

    return clusters


# ============================================================================
# 4. PROFILE REGISTRY  (the "list of profiles and characters")
# ============================================================================

@dataclass
class ProfileEntry:
    """One entry in the profile registry."""
    profile_name: str
    profile_description: str
    members: List[Tuple[str, float]]           # (character, score)
    partial_members: List[Tuple[str, float]]   # (character, score)
    criteria_summary: str


def build_profile_registry(
    assignments: Dict[str, List[ArchetypeMatch]],
) -> List[ProfileEntry]:
    """
    Build the final profile registry: a list of profiles with their
    member characters.  This is the answer to "Is there a list of
    profiles and associated characters?"
    """
    # Invert the assignments: archetype -> characters
    profile_map: Dict[str, Dict] = {}
    for archetype in ARCHETYPES:
        profile_map[archetype.name] = {
            "archetype": archetype,
            "members": [],
            "partial": [],
        }

    for char_name, matches in assignments.items():
        for match in matches:
            aname = match.archetype.name
            if match.is_member:
                profile_map[aname]["members"].append((char_name, match.score))
            elif match.is_partial:
                profile_map[aname]["partial"].append((char_name, match.score))

    registry = []
    for aname, data in profile_map.items():
        arch = data["archetype"]
        members = sorted(data["members"], key=lambda x: x[1], reverse=True)
        partial = sorted(data["partial"], key=lambda x: x[1], reverse=True)
        registry.append(ProfileEntry(
            profile_name=arch.name,
            profile_description=arch.description,
            members=members,
            partial_members=partial,
            criteria_summary=arch.describe_criteria(),
        ))

    return registry
