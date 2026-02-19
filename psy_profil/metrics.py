"""
metrics.py - Extended psycholinguistic metrics for character dialogue.

Computes four metric families per character:
  1. Lexical   - vocabulary richness, word length, hapax legomena
  2. Syntactic - sentence length, questions, exclamations, fragments
  3. Sentiment - VADER positive/negative/neutral/compound
  4. LIWC-like - psychological category word counts (custom dictionaries)
  5. POS       - part-of-speech distribution (noun/verb/adj/adverb %)

Every metric is normalised to [0, 1] for cross-character comparison.
"""

import re
import collections
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer


# ---------------------------------------------------------------------------
# NLTK bootstrap (silent)
# ---------------------------------------------------------------------------

_NLTK_RESOURCES = [
    "punkt", "punkt_tab", "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng", "vader_lexicon",
    "wordnet", "stopwords",
]


def ensure_nltk():
    for r in _NLTK_RESOURCES:
        nltk.download(r, quiet=True)


# ---------------------------------------------------------------------------
# LIWC-like dictionaries  (inspired by LIWC2015 categories)
# ---------------------------------------------------------------------------
# These are curated word-stem lists.  A word matches if it *starts with*
# any stem in the list (e.g. "destroy" matches stem "destro").

LIWC_CATEGORIES: Dict[str, List[str]] = {
    "positive_emotion": [
        "love", "nice", "sweet", "happy", "good", "great", "fine", "hope",
        "beauti", "enjoy", "wonder", "excellen", "perfect", "best", "better",
        "amaz", "safe", "trust", "care", "kind", "warm", "bright", "calm",
        "peace", "friend", "laugh", "smile", "pleas", "thank", "grate",
    ],
    "negative_emotion": [
        "hate", "bad", "wrong", "ugly", "stupid", "terrib", "horrib",
        "awful", "worst", "worse", "kill", "die", "dead", "death", "hurt",
        "pain", "damn", "hell", "sick", "dark", "cold", "fear", "angry",
        "sad", "scare", "danger", "threat", "suffer", "misera", "cruel",
    ],
    "anger": [
        "hate", "kill", "fight", "angry", "rage", "fury", "mad", "damn",
        "hell", "bastard", "stupid", "idiot", "fool", "shut", "destroy",
        "attack", "threat", "scream", "punch", "smash", "crush", "bitch",
    ],
    "anxiety": [
        "worr", "fear", "afraid", "scare", "nervous", "panic", "danger",
        "threat", "risk", "careful", "trouble", "alarm", "dread", "terror",
        "tense", "anxious", "uneasy",
    ],
    "sadness": [
        "sad", "cry", "tear", "alone", "lonely", "lost", "miss", "sorry",
        "regret", "grief", "mourn", "depress", "empty", "broken", "hopeless",
        "disappoint", "abandon",
    ],
    "social": [
        "we", "us", "our", "friend", "partner", "team", "together", "people",
        "family", "group", "trust", "help", "share", "join", "talk", "tell",
        "ask", "said", "listen", "meet",
    ],
    "cognitive": [
        "think", "know", "believe", "understand", "realiz", "wonder",
        "suppose", "guess", "figure", "reason", "logic", "consider",
        "analyz", "decide", "remember", "forget", "learn", "meaning",
        "cause", "because", "maybe", "perhaps",
    ],
    "power": [
        "control", "command", "order", "force", "power", "authorit",
        "demand", "boss", "lead", "rule", "dominat", "strong", "weapon",
        "gun", "fight", "obey", "submit", "weak",
    ],
    "achievement": [
        "win", "success", "accomplish", "achiev", "earn", "gain", "best",
        "work", "goal", "finish", "complet", "mission", "target", "done",
        "built", "made", "creat",
    ],
    "risk_danger": [
        "danger", "risk", "threat", "fire", "gun", "weapon", "fight",
        "kill", "damage", "destroy", "crash", "explos", "attack", "escape",
        "chase", "run", "shoot", "blood", "wound", "bullet",
    ],
    "certainty": [
        "always", "never", "absolute", "certain", "definit", "sure",
        "exact", "clearly", "obvious", "undoubt", "complet", "total",
    ],
    "tentativeness": [
        "maybe", "perhaps", "might", "could", "possibly", "sometimes",
        "almost", "somewhat", "seem", "appear", "sort of", "kind of",
        "guess", "suppose", "probably",
    ],
}


def _liwc_match(word: str, stems: List[str]) -> bool:
    """Check if a lowercased word starts with any stem in the list."""
    for stem in stems:
        if word.startswith(stem):
            return True
    return False


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------

@dataclass
class CharacterMetrics:
    """All computed metrics for a single character."""
    name: str
    word_count: int = 0
    sentence_count: int = 0
    warning: str | None = None  # set when sample is too small

    # Lexical
    avg_word_length: float = 0.0
    type_token_ratio: float = 0.0       # vocabulary richness
    hapax_ratio: float = 0.0            # words appearing exactly once

    # Syntactic
    avg_sentence_length: float = 0.0    # words per sentence
    question_ratio: float = 0.0         # % sentences that are questions
    exclamation_ratio: float = 0.0      # % sentences with !
    fragment_ratio: float = 0.0         # % sentences < 4 words (clipped speech)

    # Sentiment (VADER)
    sentiment_positive: float = 0.0
    sentiment_negative: float = 0.0
    sentiment_neutral: float = 0.0
    sentiment_compound: float = 0.0     # overall -1..+1

    # POS distribution (%)
    noun_pct: float = 0.0
    verb_pct: float = 0.0
    adj_pct: float = 0.0
    adv_pct: float = 0.0

    # LIWC-like category scores (% of words in each category)
    liwc: Dict[str, float] = field(default_factory=dict)

    # Top keywords (for interpretive evidence)
    top_keywords: List[Tuple[str, int]] = field(default_factory=list)

    # Raw dialogue (for Gemini interpretation)
    raw_dialogue: str = ""

    def feature_vector_names(self) -> List[str]:
        """Return ordered list of feature names for the numeric vector."""
        base = [
            "avg_word_length", "type_token_ratio", "hapax_ratio",
            "avg_sentence_length", "question_ratio", "exclamation_ratio",
            "fragment_ratio",
            "sentiment_positive", "sentiment_negative", "sentiment_neutral",
            "sentiment_compound_shifted",  # shifted to 0..1
            "noun_pct", "verb_pct", "adj_pct", "adv_pct",
        ]
        liwc_names = [f"liwc_{cat}" for cat in sorted(LIWC_CATEGORIES.keys())]
        return base + liwc_names

    def feature_vector(self) -> List[float]:
        """Return numeric feature vector (all values roughly in [0, 1])."""
        base = [
            self.avg_word_length / 10.0,   # normalise ~3-8 range
            self.type_token_ratio,
            self.hapax_ratio,
            min(self.avg_sentence_length / 30.0, 1.0),
            self.question_ratio,
            self.exclamation_ratio,
            self.fragment_ratio,
            self.sentiment_positive,
            self.sentiment_negative,
            self.sentiment_neutral,
            (self.sentiment_compound + 1.0) / 2.0,  # shift -1..+1 -> 0..1
            self.noun_pct,
            self.verb_pct,
            self.adj_pct,
            self.adv_pct,
        ]
        liwc_vals = [
            self.liwc.get(cat, 0.0) for cat in sorted(LIWC_CATEGORIES.keys())
        ]
        return base + liwc_vals


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

MIN_WORDS_FOR_RELIABLE = 30  # below this, results get a warning


def compute_metrics(name: str, dialogue_lines: List[str]) -> CharacterMetrics:
    """
    Compute all psycholinguistic metrics for one character's dialogue.
    """
    m = CharacterMetrics(name=name)

    # Join all dialogue into a single text
    full_text = " ".join(dialogue_lines)
    m.raw_dialogue = full_text

    if not full_text.strip():
        m.warning = "No dialogue found"
        return m

    # --- Tokenization ---
    sentences = sent_tokenize(full_text)
    words_raw = word_tokenize(full_text.lower())
    # Keep only alphabetic tokens
    words = [w for w in words_raw if w.isalpha() and len(w) >= 2]

    m.sentence_count = len(sentences)
    m.word_count = len(words)

    if m.word_count < MIN_WORDS_FOR_RELIABLE:
        m.warning = (
            f"Small sample ({m.word_count} words). "
            f"Results may be unreliable. Recommend >= {MIN_WORDS_FOR_RELIABLE} words."
        )

    if m.word_count == 0:
        return m

    # --- 1. Lexical Metrics ---
    m.avg_word_length = sum(len(w) for w in words) / len(words)

    unique_words = set(words)
    m.type_token_ratio = len(unique_words) / len(words)

    freq = collections.Counter(words)
    hapax = sum(1 for w, c in freq.items() if c == 1)
    m.hapax_ratio = hapax / len(unique_words) if unique_words else 0.0

    # --- 2. Syntactic Metrics ---
    sent_word_counts = []
    questions = 0
    exclamations = 0
    fragments = 0

    for sent in sentences:
        sw = [w for w in word_tokenize(sent.lower()) if w.isalpha()]
        sent_word_counts.append(len(sw))
        stripped = sent.strip()
        if stripped.endswith("?"):
            questions += 1
        if stripped.endswith("!"):
            exclamations += 1
        if len(sw) < 4:
            fragments += 1

    m.avg_sentence_length = (
        sum(sent_word_counts) / len(sent_word_counts) if sent_word_counts else 0.0
    )
    m.question_ratio = questions / len(sentences) if sentences else 0.0
    m.exclamation_ratio = exclamations / len(sentences) if sentences else 0.0
    m.fragment_ratio = fragments / len(sentences) if sentences else 0.0

    # --- 3. Sentiment (VADER) ---
    sia = SentimentIntensityAnalyzer()
    pos_sum = neg_sum = neu_sum = comp_sum = 0.0
    for sent in sentences:
        vs = sia.polarity_scores(sent)
        pos_sum += vs["pos"]
        neg_sum += vs["neg"]
        neu_sum += vs["neu"]
        comp_sum += vs["compound"]

    n_sent = max(len(sentences), 1)
    m.sentiment_positive = pos_sum / n_sent
    m.sentiment_negative = neg_sum / n_sent
    m.sentiment_neutral = neu_sum / n_sent
    m.sentiment_compound = comp_sum / n_sent

    # --- 4. POS Distribution ---
    try:
        tagged = nltk.pos_tag(words)
    except LookupError:
        tagged = [(w, "NN") for w in words]

    pos_counts = collections.Counter()
    for _, tag in tagged:
        if tag.startswith("NN"):
            pos_counts["noun"] += 1
        elif tag.startswith("VB"):
            pos_counts["verb"] += 1
        elif tag.startswith("JJ"):
            pos_counts["adj"] += 1
        elif tag.startswith("RB"):
            pos_counts["adv"] += 1

    total_pos = sum(pos_counts.values()) or 1
    m.noun_pct = pos_counts["noun"] / total_pos
    m.verb_pct = pos_counts["verb"] / total_pos
    m.adj_pct = pos_counts["adj"] / total_pos
    m.adv_pct = pos_counts["adv"] / total_pos

    # --- 5. LIWC-like Categories ---
    for cat_name, stems in LIWC_CATEGORIES.items():
        matches = sum(1 for w in words if _liwc_match(w, stems))
        m.liwc[cat_name] = matches / len(words)

    # --- 6. Top Keywords (lemmatised content words) ---
    lemmatizer = WordNetLemmatizer()
    try:
        stop_words = set(nltk.corpus.stopwords.words("english"))
    except LookupError:
        stop_words = set()

    content_words = []
    for word, tag in tagged:
        if word in stop_words or len(word) < 3:
            continue
        if tag.startswith(("NN", "JJ", "VB")):
            pos_code = "n" if tag.startswith("NN") else "v" if tag.startswith("VB") else "a"
            try:
                lemma = lemmatizer.lemmatize(word, pos=pos_code)
            except Exception:
                lemma = word
            content_words.append(lemma)

    kw_freq = collections.Counter(content_words)
    m.top_keywords = kw_freq.most_common(15)

    return m
