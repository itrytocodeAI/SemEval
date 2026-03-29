"""
DimABSA2026 – Arousal Feature Extractor
=========================================
8-dimensional linguistic feature vector designed to capture
intensity/arousal signals that transformers struggle with.

Features:
  1. n_exclamation     — count of "!"
  2. n_caps_words      — ALL CAPS words (len>=2)
  3. n_intensifiers    — "very", "extremely", "absolutely", etc.
  4. n_emotion_words   — NRC/custom emotion lexicon matches
  5. avg_word_length   — proxy for formality
  6. sentence_length   — total word count
  7. n_question_marks  — count of "?"
  8. n_negations       — "not", "never", "no", "don't", etc.
"""

import re
import torch
import torch.nn as nn

# ── Lexicons ──────────────────────────────────────────────────────────────
INTENSIFIERS = {
    "very", "extremely", "absolutely", "incredibly", "really", "so",
    "totally", "utterly", "highly", "deeply", "terribly", "awfully",
    "remarkably", "exceptionally", "especially", "particularly",
    "quite", "rather", "seriously", "genuinely", "truly", "completely",
    "perfectly", "enormously", "massively", "somewhat", "fairly",
}

NEGATIONS = {
    "not", "never", "no", "don't", "doesn't", "didn't", "won't",
    "wouldn't", "shouldn't", "couldn't", "can't", "cannot", "nor",
    "neither", "hardly", "barely", "scarcely", "nothing", "nobody",
    "nowhere", "isn't", "aren't", "wasn't", "weren't", "haven't",
}

EMOTION_WORDS = {
    # High arousal positive
    "amazing", "fantastic", "incredible", "awesome", "thrilling",
    "exciting", "ecstatic", "magnificent", "spectacular", "phenomenal",
    "stunning", "breathtaking", "exhilarating", "electrifying",
    # High arousal negative
    "terrible", "horrible", "awful", "disgusting", "furious",
    "outrageous", "appalling", "dreadful", "atrocious", "abysmal",
    "infuriating", "enraging", "revolting", "horrifying",
    # Medium arousal
    "love", "hate", "angry", "scared", "happy", "sad", "worried",
    "anxious", "frustrated", "delighted", "pleased", "annoyed",
    "disappointed", "surprised", "shocked", "thrilled", "miserable",
    # Low arousal
    "boring", "dull", "bland", "mediocre", "plain", "ordinary",
    "calm", "peaceful", "relaxed", "quiet", "gentle", "soothing",
}


def extract_features(text: str) -> list:
    """Extract 8-dim arousal-relevant linguistic feature vector from text."""
    words = text.split()
    lower_words = [w.lower().strip(".,!?;:'\"()-") for w in words]

    # 1. Exclamation marks
    n_excl = text.count("!")

    # 2. ALL CAPS words (length >= 2, skip single letters)
    n_caps = sum(1 for w in words if w.isupper() and len(w) >= 2)

    # 3. Intensifiers
    n_intens = sum(1 for w in lower_words if w in INTENSIFIERS)

    # 4. Emotion words
    n_emotion = sum(1 for w in lower_words if w in EMOTION_WORDS)

    # 5. Average word length
    avg_wlen = sum(len(w) for w in words) / max(len(words), 1)

    # 6. Sentence length (word count)
    sent_len = len(words)

    # 7. Question marks
    n_quest = text.count("?")

    # 8. Negations
    n_neg = sum(1 for w in lower_words if w in NEGATIONS)

    return [n_excl, n_caps, n_intens, n_emotion, avg_wlen, sent_len, n_quest, n_neg]


def extract_features_batch(texts: list) -> torch.Tensor:
    """Extract features for a batch of texts. Returns (batch, 8) tensor."""
    feats = [extract_features(t) for t in texts]
    return torch.tensor(feats, dtype=torch.float32)


class ArousalFeatureNorm(nn.Module):
    """LayerNorm wrapper for the 8-dim feature vector.
    
    Critical: Raw counts (0-50+) must be normalized before
    concatenation with DeBERTa embeddings (-1 to 1).
    """
    def __init__(self, n_features=8):
        super().__init__()
        self.norm = nn.LayerNorm(n_features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch, 8) → normalized (batch, 8)"""
        return self.norm(features)


# ── Quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_texts = [
        "The food was ABSOLUTELY AMAZING!!! Best restaurant ever!",
        "Decent pasta. Nothing special, quite ordinary.",
        "TERRIBLE service!!! Never coming back! SO ANGRY!!!",
        "The soup was okay, not bad but not great either.",
    ]
    for t in test_texts:
        f = extract_features(t)
        print(f"  {f}  ←  {t[:50]}")
