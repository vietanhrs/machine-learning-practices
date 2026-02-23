"""
Exercise 01: N-gram Language Model
===================================
Build an N-gram language model from scratch with add-k smoothing.
You will implement tokenization, n-gram counting, probability estimation,
text generation, and perplexity evaluation.

Learning Goals:
    - Understand how N-gram models estimate conditional probabilities
    - Implement add-k smoothing to handle unseen n-grams
    - Evaluate models using perplexity
    - Observe how n-gram order affects generation quality
"""

import math
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Sample corpus — feel free to replace with a larger text or nltk corpus
# ---------------------------------------------------------------------------
SAMPLE_CORPUS = """
To be or not to be that is the question whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune or to take arms against a sea of troubles
and by opposing end them to die to sleep no more and by a sleep to say we end
the heartache and the thousand natural shocks that flesh is heir to tis a consummation
devoutly to be wished to die to sleep to sleep perchance to dream ay there is the rub
for in that sleep of death what dreams may come when we have shuffled off this mortal
coil must give us pause there is the respect that makes calamity of so long life
"""


# ---------------------------------------------------------------------------
# 1. Tokenization
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """
    Convert raw text into a list of lowercase tokens.

    Args:
        text: Raw input string.

    Returns:
        List of lowercase word tokens with punctuation removed.

    TODO:
        1. Convert text to lowercase.
        2. Use re.findall or re.sub to split on whitespace and punctuation.
           Hint: re.findall(r"[a-z]+", text.lower()) extracts only alphabetic tokens.
        3. Filter out empty strings.
        4. Return the list of tokens.
    """
    raise NotImplementedError("TODO: implement tokenize()")


# ---------------------------------------------------------------------------
# 2. N-gram Counting
# ---------------------------------------------------------------------------

def build_ngram_counts(tokens: List[str], n: int) -> Dict[Tuple, int]:
    """
    Count occurrences of every n-gram in the token list.

    Args:
        tokens: List of word tokens.
        n:      The n in n-gram (1 for unigram, 2 for bigram, etc.)

    Returns:
        Dictionary mapping n-gram tuples to their counts.
        Example for n=2: {('to', 'be'): 5, ('be', 'or'): 1, ...}

    TODO:
        1. Iterate over tokens with a sliding window of size n.
           For position i, the n-gram is tuple(tokens[i : i+n]).
        2. Count each n-gram using a Counter or defaultdict(int).
        3. Return the count dictionary.

    Hint:
        - For n=1 (unigram), just count individual words.
        - zip(*[tokens[i:] for i in range(n)]) is a clean way to generate all n-grams.
    """
    raise NotImplementedError("TODO: implement build_ngram_counts()")


# ---------------------------------------------------------------------------
# 3. N-gram Model (conditional probabilities with add-k smoothing)
# ---------------------------------------------------------------------------

def build_ngram_model(
    tokens: List[str], n: int, smoothing_k: float = 1.0
) -> Dict[Tuple, Dict[str, float]]:
    """
    Build a conditional probability table for an n-gram model with add-k smoothing.

    Args:
        tokens:     List of word tokens.
        n:          The n in n-gram.
        smoothing_k: The smoothing constant k (default 1.0 = Laplace smoothing).

    Returns:
        A nested dictionary: model[context_tuple][next_word] = probability.
        For n=2 (bigram): model[('to',)]['be'] = P('be' | 'to')

    TODO:
        1. Build a vocabulary set from tokens.
        2. Count all n-grams using build_ngram_counts(tokens, n).
        3. Count all (n-1)-grams (the context) using build_ngram_counts(tokens, n-1).
           For unigrams (n=1), the context is empty tuple () and the denominator is len(tokens).
        4. For each n-gram (w_1, ..., w_{n-1}, w_n):
             context = (w_1, ..., w_{n-1})
             P(w_n | context) = (count(context + w_n) + k) / (count(context) + k * |V|)
        5. Store in a nested dict: model[context][w_n] = probability.
        6. Return the model.

    Note:
        Add-k smoothing formula:
            P(w | context) = (C(context, w) + k) / (C(context) + k * |V|)
        where |V| is the vocabulary size.
    """
    raise NotImplementedError("TODO: implement build_ngram_model()")


# ---------------------------------------------------------------------------
# 4. Predict Next Word
# ---------------------------------------------------------------------------

def predict_next_word(
    context: List[str],
    model: Dict[Tuple, Dict[str, float]],
    n: int,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Predict the top-k most likely next words given a context.

    Args:
        context: List of preceding words (the history).
        model:   N-gram model from build_ngram_model().
        n:       The n in n-gram (must match model).
        top_k:   Number of top predictions to return.

    Returns:
        List of (word, probability) tuples sorted by probability descending.
        Example: [('be', 0.25), ('sleep', 0.15), ...]

    TODO:
        1. Extract the relevant context window: the last (n-1) words from context.
           For n=1 (unigram), context_key = ().
           For n=2 (bigram), context_key = (context[-1],).
        2. Convert context_key to a tuple.
        3. Look up context_key in the model dictionary.
        4. If the context is not found, return an empty list (or handle gracefully).
        5. Sort the word-probability pairs by probability in descending order.
        6. Return the top_k results.
    """
    raise NotImplementedError("TODO: implement predict_next_word()")


# ---------------------------------------------------------------------------
# 5. Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(
    tokens: List[str],
    model: Dict[Tuple, Dict[str, float]],
    n: int,
) -> float:
    """
    Compute perplexity of the model on a sequence of tokens.

    Args:
        tokens: Test token sequence.
        model:  N-gram model from build_ngram_model().
        n:      The n in n-gram.

    Returns:
        Perplexity score (lower is better).

    TODO:
        1. For each token w_i (starting at index n-1 to have enough context):
             a. Extract context = tuple(tokens[i-n+1 : i])
                For n=1, context = ().
             b. Look up P(w_i | context) from the model.
             c. If probability is 0 or not found, use a very small floor (e.g., 1e-10)
                to avoid log(0).
             d. Accumulate: log_sum += math.log(probability)
        2. Compute average negative log-likelihood: nll = -log_sum / N
           where N is the number of tokens evaluated.
        3. Return math.exp(nll).

    Formula:
        PP(W) = exp( -1/N * Σ log P(w_i | context_i) )
    """
    raise NotImplementedError("TODO: implement compute_perplexity()")


# ---------------------------------------------------------------------------
# 6. Text Generation
# ---------------------------------------------------------------------------

def generate_text(
    model: Dict[Tuple, Dict[str, float]],
    n: int,
    seed_words: List[str],
    max_words: int = 50,
    temperature: float = 1.0,
) -> str:
    """
    Generate text using the n-gram model.

    Args:
        model:      N-gram model from build_ngram_model().
        n:          The n in n-gram.
        seed_words: Starting words to condition generation on.
        max_words:  Maximum number of words to generate.
        temperature: Sampling temperature (1.0 = unmodified; <1 = sharper; >1 = more random).

    Returns:
        Generated text as a single string.

    TODO:
        1. Start with a copy of seed_words as the generated sequence.
        2. Repeat until max_words is reached:
             a. Extract context = last (n-1) words from the generated sequence.
                For n=1, context = ().
             b. Look up context in model. If not found, break (no context match).
             c. Get the word-probability dict for this context.
             d. Apply temperature scaling:
                - Get all words and their log-probabilities: log_p = log(p) / temperature
                - Convert back to probabilities with softmax (or just normalize exps):
                  scaled_probs = [exp(log_p) for log_p in log_probs]
                  normalized = [p / sum(scaled_probs) for p in scaled_probs]
             e. Sample the next word using random.choices(words, weights=normalized).
             f. Append the sampled word to the sequence.
        3. Return ' '.join(generated_sequence).

    Hint:
        - For greedy decoding (temperature ≈ 0), just take the argmax.
        - random.choices(population, weights=weights, k=1)[0] samples one item.
    """
    raise NotImplementedError("TODO: implement generate_text()")


# ---------------------------------------------------------------------------
# 7. Compare N-gram Orders
# ---------------------------------------------------------------------------

def compare_ngram_orders(
    corpus: str,
    test_tokens: Optional[List[str]] = None,
    max_n: int = 3,
) -> Dict[int, float]:
    """
    Train n-gram models for n=1, 2, ..., max_n and compare their perplexity on test data.

    Args:
        corpus:      Training text corpus (raw string).
        test_tokens: Tokens to evaluate on. If None, use last 20% of corpus tokens.
        max_n:       Maximum n-gram order to evaluate.

    Returns:
        Dictionary mapping n → perplexity.

    TODO:
        1. Tokenize the corpus.
        2. If test_tokens is None, split tokens: 80% train / 20% test.
        3. For n in range(1, max_n + 1):
             a. Build the n-gram model on training tokens.
             b. Compute perplexity on test tokens.
             c. Store result in results[n].
        4. Print a summary table showing n → perplexity.
        5. Return the results dictionary.
    """
    raise NotImplementedError("TODO: implement compare_ngram_orders()")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Demonstrate the n-gram language model on the sample corpus.

    Steps:
        1. Tokenize the corpus.
        2. Build bigram and trigram models.
        3. Show top-5 predictions for a sample context.
        4. Generate text at different temperatures.
        5. Compare perplexity across n-gram orders.
    """
    print("=" * 60)
    print("N-gram Language Model Demo")
    print("=" * 60)

    tokens = tokenize(SAMPLE_CORPUS)
    print(f"\nCorpus: {len(tokens)} tokens | Vocabulary: {len(set(tokens))} unique words\n")

    # --- Build models ---
    bigram_model  = build_ngram_model(tokens, n=2, smoothing_k=1.0)
    trigram_model = build_ngram_model(tokens, n=3, smoothing_k=0.1)

    # --- Predict next word ---
    print("--- Bigram predictions after 'to' ---")
    predictions = predict_next_word(["to"], bigram_model, n=2, top_k=5)
    for word, prob in predictions:
        print(f"  {word:<15} {prob:.4f}")

    # --- Generate text ---
    print("\n--- Generated text (bigram, temperature=1.0) ---")
    print(generate_text(bigram_model, n=2, seed_words=["to", "be"], max_words=30))

    print("\n--- Generated text (bigram, temperature=0.5) ---")
    print(generate_text(bigram_model, n=2, seed_words=["to", "be"], max_words=30, temperature=0.5))

    # --- Compare perplexity across n-gram orders ---
    print("\n--- Perplexity comparison across n-gram orders ---")
    results = compare_ngram_orders(SAMPLE_CORPUS, max_n=3)
    for n, pp in sorted(results.items()):
        print(f"  n={n}  perplexity={pp:.2f}")


if __name__ == "__main__":
    main()
