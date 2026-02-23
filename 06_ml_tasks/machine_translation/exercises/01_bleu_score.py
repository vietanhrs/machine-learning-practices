"""
Exercise 01: BLEU Score Implementation
=======================================
Implement the BLEU (Bilingual Evaluation Understudy) score from scratch.
BLEU is the standard automatic evaluation metric for machine translation,
measuring n-gram overlap between hypothesis and reference translations.

Learning Goals:
    - Understand clipped n-gram precision and why clipping is needed
    - Implement the brevity penalty (penalizes short hypotheses)
    - Compute corpus-level BLEU across multiple sentence pairs
    - Observe how BLEU captures translation quality (and its limitations)

Reference:
    Papineni et al. (2002) "BLEU: a Method for Automatic Evaluation of Machine Translation"
    https://aclanthology.org/P02-1040.pdf

Requirements:
    pip install numpy  (no other dependencies needed)
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Tokenization
# ---------------------------------------------------------------------------

def tokenize(sentence: str) -> List[str]:
    """
    Tokenize a sentence into a list of lowercase words.

    Args:
        sentence: Input string.

    Returns:
        List of lowercase word tokens.

    TODO:
        1. Convert to lowercase.
        2. Split on whitespace and remove punctuation.
           Simple approach: sentence.lower().split()
           Better: use re.findall(r'\b\w+\b', sentence.lower())
        3. Return the token list.
    """
    raise NotImplementedError("TODO: implement tokenize()")


# ---------------------------------------------------------------------------
# 2. N-gram Counting
# ---------------------------------------------------------------------------

def ngram_counts(tokens: List[str], n: int) -> Counter:
    """
    Count all n-grams in a token list.

    Args:
        tokens: List of word tokens.
        n:      N-gram order (1 for unigram, 2 for bigram, etc.).

    Returns:
        Counter mapping n-gram tuples to their counts.

    TODO:
        1. Generate all n-grams from tokens:
               ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        2. Return Counter(ngrams).
    """
    raise NotImplementedError("TODO: implement ngram_counts()")


# ---------------------------------------------------------------------------
# 3. Clipped Precision
# ---------------------------------------------------------------------------

def clipped_precision(
    hypothesis: List[str],
    references: List[List[str]],
    n: int,
) -> float:
    """
    Compute the modified (clipped) n-gram precision for a single hypothesis
    against one or more reference translations.

    Clipping: each n-gram in the hypothesis is counted at most as many times
    as it appears in the best-matching (most favorable) reference.

    Args:
        hypothesis:  Token list of the model's translation.
        references:  List of token lists (one per reference translation).
        n:           N-gram order.

    Returns:
        Clipped n-gram precision as a float in [0, 1].
        Returns 0.0 if there are no n-grams in the hypothesis.

    TODO:
        1. Count n-grams in the hypothesis: hyp_counts = ngram_counts(hypothesis, n).
        2. For each reference, count n-grams: ref_counts = ngram_counts(ref, n).
        3. For each n-gram in hyp_counts:
               max_ref_count = max(ref_counts.get(ngram, 0) for ref in references)
               clipped_count = min(hyp_counts[ngram], max_ref_count)
        4. clipped_total = sum of clipped counts for all n-grams in hypothesis.
        5. hyp_total = sum of hyp_counts (total n-gram count in hypothesis).
        6. Return clipped_total / hyp_total if hyp_total > 0 else 0.0.

    Example:
        hypothesis = ['the', 'cat', 'the', 'cat']
        reference  = ['the', 'cat', 'sat']
        Unigram counts in hyp: {'the': 2, 'cat': 2}
        Max ref counts:         'the' → 1, 'cat' → 1
        Clipped:                'the' → min(2,1)=1, 'cat' → min(2,1)=1
        Clipped total = 2, Hyp total = 4 → precision = 2/4 = 0.5
    """
    raise NotImplementedError("TODO: implement clipped_precision()")


# ---------------------------------------------------------------------------
# 4. Brevity Penalty
# ---------------------------------------------------------------------------

def brevity_penalty(
    hypothesis: List[str],
    references: List[List[str]],
) -> float:
    """
    Compute the brevity penalty for a single hypothesis.

    The brevity penalty penalizes hypotheses that are shorter than the reference.
    It equals 1.0 when the hypothesis is longer than the closest-length reference.

    Args:
        hypothesis:  Token list of the model's translation.
        references:  List of token lists (reference translations).

    Returns:
        Brevity penalty as a float in (0, 1].

    TODO:
        1. Compute hyp_len = len(hypothesis).
        2. Find the closest reference length:
               ref_closest_len = min(len(ref) for ref in references,
                                     key=lambda ref_len: abs(ref_len - hyp_len))
           In case of a tie (two references equidistant from hyp_len),
           use the shorter reference length.
        3. If hyp_len >= ref_closest_len: return 1.0 (no penalty).
        4. Else: return math.exp(1 - ref_closest_len / hyp_len).

    Note:
        When hyp_len == 0, return 0.0 (undefined edge case — empty hypothesis).
    """
    raise NotImplementedError("TODO: implement brevity_penalty()")


# ---------------------------------------------------------------------------
# 5. BLEU Score (Sentence-Level)
# ---------------------------------------------------------------------------

def bleu_score(
    hypothesis: List[str],
    references: List[List[str]],
    max_n: int = 4,
    weights: Optional[List[float]] = None,
) -> float:
    """
    Compute the BLEU score for a single hypothesis sentence.

    Args:
        hypothesis:  Token list of the model's translation.
        references:  List of token lists (one or more reference translations).
        max_n:       Maximum n-gram order to use (standard BLEU-4 uses max_n=4).
        weights:     Weights for each n-gram order. Defaults to uniform [1/max_n]*max_n.

    Returns:
        BLEU score as a float in [0, 1].

    TODO:
        1. If weights is None, set weights = [1.0 / max_n] * max_n.
        2. Compute the brevity penalty: bp = brevity_penalty(hypothesis, references).
        3. For each n from 1 to max_n:
               prec_n = clipped_precision(hypothesis, references, n)
               if prec_n == 0:
                   → BLEU is 0 (log(0) = -inf → BLEU = 0)
                   → return 0.0 immediately (or add a small epsilon to avoid log(0))
        4. Compute the weighted sum of log-precisions:
               log_avg = Σ_{n=1}^{max_n} weights[n-1] * log(prec_n)
        5. Return bp * exp(log_avg).

    Formula:
        BLEU = BP * exp(Σ_n w_n * log(p_n))

    Note:
        Standard practice adds a small epsilon (e.g., 1e-10) to precision before
        taking log, to handle 0 precision without crashing.
        OR: return 0.0 immediately if any precision is 0.
    """
    raise NotImplementedError("TODO: implement bleu_score()")


# ---------------------------------------------------------------------------
# 6. Corpus-Level BLEU
# ---------------------------------------------------------------------------

def corpus_bleu(
    hypotheses: List[List[str]],
    references_list: List[List[List[str]]],
    max_n: int = 4,
) -> float:
    """
    Compute corpus-level BLEU score.

    Corpus BLEU aggregates counts across all sentences before computing precision,
    rather than averaging sentence-level BLEU scores. This is the standard method.

    Args:
        hypotheses:       List of hypothesis token lists (one per sentence).
        references_list:  List of reference sets. references_list[i] is a list of
                          reference token lists for hypothesis i.
        max_n:            Maximum n-gram order.

    Returns:
        Corpus-level BLEU score in [0, 1].

    TODO:
        1. For each n from 1 to max_n:
               Accumulate total_clipped[n] = sum of clipped n-gram counts across all hypotheses.
               Accumulate total_hyp[n] = sum of total n-gram counts in all hypotheses.
        2. Accumulate total hyp_len and closest ref_len across all sentences (for BP).
        3. Compute corpus BP:
               bp = 1.0 if total_hyp_len >= total_ref_len
               else exp(1 - total_ref_len / total_hyp_len)
        4. Compute corpus-level precisions:
               prec_n = total_clipped[n] / total_hyp[n]
        5. Return bp * exp(Σ_n (1/max_n) * log(prec_n)).

    Note:
        Corpus BLEU is generally more reliable than averaging sentence BLEU scores,
        especially for short sentences (where sentence BLEU is very noisy).
    """
    raise NotImplementedError("TODO: implement corpus_bleu()")


# ---------------------------------------------------------------------------
# 7. Compare Translations
# ---------------------------------------------------------------------------

def compare_translations(
    source: str,
    references: List[str],
    candidates: List[Tuple[str, str]],
    max_n: int = 4,
) -> None:
    """
    Rank candidate translations by BLEU score and display results.

    Args:
        source:     Source sentence (for display only).
        references: List of reference translation strings.
        candidates: List of (name, translation) tuples to evaluate.
        max_n:      Maximum n-gram order for BLEU.

    TODO:
        1. Tokenize all references and candidate translations.
        2. For each (name, translation) in candidates:
               score = bleu_score(tokenize(translation),
                                  [tokenize(ref) for ref in references],
                                  max_n=max_n)
        3. Sort candidates by BLEU score descending.
        4. Print a formatted table showing:
               Rank | Name | BLEU | Translation
    """
    raise NotImplementedError("TODO: implement compare_translations()")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Demonstrate BLEU score computation on example translations.

    Examples show:
        - Perfect translation (BLEU = 1.0)
        - Near-perfect translation (one word different)
        - Partially correct translation
        - Completely wrong translation
        - Short hypothesis (brevity penalty effect)
        - Corpus-level BLEU
    """
    print("=" * 65)
    print("BLEU Score Implementation Demo")
    print("=" * 65)

    # --- Reference sentence ---
    reference = "the cat sat on the mat"
    references = [reference]
    ref_tokens = [tokenize(reference)]

    candidates = [
        ("Perfect",           "the cat sat on the mat"),
        ("Near-perfect",      "the cat sat on a mat"),
        ("Synonyms",          "the kitten rested on the carpet"),
        ("Reordered words",   "the mat the cat sat on"),
        ("Partial (4 words)", "the cat sat on"),
        ("Completely wrong",  "a dog ran through the park"),
        ("Empty-ish (1 word)","the"),
    ]

    print(f"\nReference: '{reference}'\n")
    print(f"{'Candidate':<25} {'BLEU-4':>8} {'BLEU-1':>8} {'BP':>8}")
    print("-" * 55)
    for name, cand in candidates:
        hyp_tokens = tokenize(cand)
        b4 = bleu_score(hyp_tokens, ref_tokens, max_n=4)
        b1 = bleu_score(hyp_tokens, ref_tokens, max_n=1)
        bp = brevity_penalty(hyp_tokens, ref_tokens)
        print(f"{name:<25} {b4:>8.4f} {b1:>8.4f} {bp:>8.4f}")

    # --- Multiple references ---
    print("\n--- Multiple Reference Translations ---")
    multi_refs = [
        "the cat sat on the mat",
        "the feline rested upon the rug",
        "a cat was sitting on a mat",
    ]
    multi_ref_tokens = [tokenize(r) for r in multi_refs]

    for name, cand in candidates[:4]:
        hyp_tokens = tokenize(cand)
        score_single = bleu_score(hyp_tokens, ref_tokens, max_n=4)
        score_multi  = bleu_score(hyp_tokens, multi_ref_tokens, max_n=4)
        print(f"  {name:<25}  1-ref: {score_single:.4f}   3-refs: {score_multi:.4f}")

    # --- N-gram order effect ---
    print("\n--- Effect of N-gram Order ---")
    near_perfect = tokenize("the cat sat on a mat")
    for n in [1, 2, 3, 4]:
        score = bleu_score(near_perfect, ref_tokens, max_n=n)
        print(f"  BLEU-{n}: {score:.4f}")

    # --- Corpus-level BLEU ---
    print("\n--- Corpus-Level BLEU ---")
    hyp_corpus = [
        tokenize("the cat sat on the mat"),
        tokenize("the dog ran in the park"),
        tokenize("birds are flying south"),
    ]
    ref_corpus = [
        [tokenize("the cat sat on the mat")],
        [tokenize("the dog was running in the park")],
        [tokenize("birds fly south for the winter")],
    ]
    corpus_score = corpus_bleu(hyp_corpus, ref_corpus)
    print(f"  Corpus BLEU-4: {corpus_score:.4f}")

    # --- Compare translations ---
    print("\n--- Translation Ranking ---")
    compare_translations(
        source="The cat sat on the mat.",
        references=["the cat sat on the mat", "a cat was sitting on the mat"],
        candidates=[
            ("Google-like MT",     "the cat sat on the mat"),
            ("Good MT",            "the cat was sitting on the mat"),
            ("Mediocre MT",        "cat sat mat on the"),
            ("Confused MT",        "the dog slept under the table"),
        ],
    )


if __name__ == "__main__":
    main()
