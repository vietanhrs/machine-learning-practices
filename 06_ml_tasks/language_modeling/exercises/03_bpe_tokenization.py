"""
Exercise 03: Byte Pair Encoding (BPE) Tokenization
====================================================
Implement the BPE subword tokenization algorithm from scratch.
BPE starts with character-level units and iteratively merges the most
frequent adjacent symbol pairs, building up a vocabulary of subword units.

Learning Goals:
    - Understand why subword tokenization is preferred over word-level
    - Implement BPE training (iterative pair merging)
    - Apply learned merge rules to new text
    - Observe how vocabulary evolves with the number of merges

Reference:
    Sennrich et al. (2016) "Neural Machine Translation of Rare Words with
    Subword Units" — https://arxiv.org/abs/1508.07909
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Sample corpus
# ---------------------------------------------------------------------------
SAMPLE_CORPUS = [
    "low lower lowest",
    "new newer newest",
    "wide wider widest",
    "old older oldest",
    "cool cooler coolest",
    "fast faster fastest",
    "slow slower slowest",
    "bright brighter brightest",
    "dark darker darkest",
    "light lighter lightest",
]


# ---------------------------------------------------------------------------
# 1. Build Initial Vocabulary
# ---------------------------------------------------------------------------

def get_vocab(corpus: List[str]) -> Dict[Tuple[str, ...], int]:
    """
    Build the initial character-level vocabulary from the corpus.

    Each word is split into individual characters, with a special end-of-word
    marker '</w>' appended to the last character. The vocabulary maps
    character-tuple representations of words to their frequency.

    Args:
        corpus: List of strings (sentences or lines of text).

    Returns:
        Dictionary mapping a tuple of characters (with '</w>') to word frequency.

        Example:
            "low" → ('l', 'o', 'w', '</w>') : 3
            "lower" → ('l', 'o', 'w', 'e', 'r', '</w>') : 2

    TODO:
        1. Count word frequencies across all lines in the corpus.
           Use a Counter or defaultdict(int).
        2. For each unique word:
             a. Split into individual characters.
             b. Append '</w>' to the last character (or as a separate symbol).
                Convention: treat '</w>' as a standalone symbol at the end.
                Representation: tuple(['l', 'o', 'w', '</w>']) for "low".
        3. Build a dict: {char_tuple: word_freq}.
        4. Return the vocabulary dict.

    Hint:
        word_chars = tuple(list(word) + ['</w>'])
    """
    raise NotImplementedError("TODO: implement get_vocab()")


# ---------------------------------------------------------------------------
# 2. Pair Frequencies
# ---------------------------------------------------------------------------

def get_pair_frequencies(
    vocab: Dict[Tuple[str, ...], int]
) -> Dict[Tuple[str, str], int]:
    """
    Count the frequency of every adjacent symbol pair in the vocabulary.

    Args:
        vocab: Current vocabulary (character/subword tuples → frequency).

    Returns:
        Dictionary mapping (symbol_a, symbol_b) → total frequency across all words.

    TODO:
        1. Initialize a Counter or defaultdict(int).
        2. For each (word_tuple, freq) in vocab.items():
             For each adjacent pair (word_tuple[i], word_tuple[i+1]):
                 pair_counts[(word_tuple[i], word_tuple[i+1])] += freq
        3. Return the pair frequency dict.

    Example:
        vocab = {('l','o','w','</w>'): 3, ('l','o','w','e','r','</w>'): 2}
        → pair_freqs = {('l','o'): 5, ('o','w'): 5, ('w','</w>'): 3,
                        ('w','e'): 2, ('e','r'): 2, ('r','</w>'): 2}
    """
    raise NotImplementedError("TODO: implement get_pair_frequencies()")


# ---------------------------------------------------------------------------
# 3. Merge a Pair
# ---------------------------------------------------------------------------

def merge_pair(
    vocab: Dict[Tuple[str, ...], int],
    best_pair: Tuple[str, str],
) -> Dict[Tuple[str, ...], int]:
    """
    Merge all occurrences of the best symbol pair in the vocabulary.

    Args:
        vocab:     Current vocabulary.
        best_pair: The pair of symbols (a, b) to merge into "ab".

    Returns:
        Updated vocabulary with the pair merged everywhere it appears.

    TODO:
        1. Create a new vocabulary dict (don't modify in-place).
        2. For each (word_tuple, freq) in vocab.items():
             a. Scan the word_tuple for occurrences of best_pair = (a, b).
             b. Wherever (a, b) appears as adjacent symbols, replace with the
                single symbol a+b (string concatenation: 'l'+'o' → 'lo').
             c. Reconstruct the new word tuple with the merge applied.
        3. Store new_word_tuple → freq in the new vocab.
        4. Return the new vocab.

    Hint:
        A clean way to apply the merge:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
    """
    raise NotImplementedError("TODO: implement merge_pair()")


# ---------------------------------------------------------------------------
# 4. Train BPE
# ---------------------------------------------------------------------------

def train_bpe(
    corpus: List[str],
    num_merges: int,
) -> Tuple[Dict[Tuple[str, ...], int], List[Tuple[str, str]]]:
    """
    Train BPE by iteratively merging the most frequent symbol pair.

    Args:
        corpus:     List of strings (training corpus).
        num_merges: Number of merge operations to perform.

    Returns:
        final_vocab:  The vocabulary after all merges.
        merge_rules:  Ordered list of merge rules (each a pair tuple).
                      Order matters — apply them in sequence during tokenization.

    TODO:
        1. Initialize vocab = get_vocab(corpus).
        2. Initialize merge_rules = [].
        3. For each merge step (0 to num_merges - 1):
             a. Compute pair_freqs = get_pair_frequencies(vocab).
             b. If pair_freqs is empty, stop early (no more pairs to merge).
             c. Find the most frequent pair:
                best_pair = max(pair_freqs, key=pair_freqs.get)
             d. Merge: vocab = merge_pair(vocab, best_pair).
             e. Append best_pair to merge_rules.
             f. Optionally print: step, best_pair, pair_freqs[best_pair].
        4. Return (vocab, merge_rules).
    """
    raise NotImplementedError("TODO: implement train_bpe()")


# ---------------------------------------------------------------------------
# 5. Apply BPE to New Words
# ---------------------------------------------------------------------------

def tokenize_bpe(
    word: str,
    merge_rules: List[Tuple[str, str]],
) -> List[str]:
    """
    Tokenize a new word by applying the learned BPE merge rules in order.

    Args:
        word:        A single word (string) to tokenize.
        merge_rules: Ordered list of merge rules from train_bpe().

    Returns:
        List of subword tokens.
        Example: "lowest" → ['low', 'est', '</w>'] (after many merges)

    TODO:
        1. Split the word into characters + '</w>': list(word) + ['</w>'].
        2. Apply each merge rule in order:
             For rule (a, b):
                 Scan the current list of symbols.
                 Wherever (symbols[i], symbols[i+1]) == (a, b), merge into a+b.
        3. Return the final list of symbols (subword tokens).

    Note:
        The merge rules must be applied in the SAME ORDER as they were learned.
        Each rule may enable new merges possible in later rules.
    """
    raise NotImplementedError("TODO: implement tokenize_bpe()")


# ---------------------------------------------------------------------------
# 6. Compare Vocabulary Sizes
# ---------------------------------------------------------------------------

def compare_vocabulary_sizes(
    corpus: List[str],
    num_merges_list: Optional[List[int]] = None,
) -> Dict[int, int]:
    """
    Train BPE with different numbers of merges and compare resulting vocabulary sizes.

    Args:
        corpus:          Training corpus.
        num_merges_list: List of merge counts to compare.
                         Defaults to [0, 10, 20, 50, 100].

    Returns:
        Dictionary mapping num_merges → unique subword vocabulary size.

    TODO:
        1. For each n in num_merges_list:
             a. Run train_bpe(corpus, n) → (vocab, rules).
             b. Count the number of unique symbols across all vocab entries.
                unique_symbols = set of all symbols appearing in any word tuple in vocab.
             c. Store results[n] = len(unique_symbols).
        2. Print a summary table.
        3. Return results.

    Hint:
        unique_symbols = {symbol for word_tuple in vocab for symbol in word_tuple}
    """
    raise NotImplementedError("TODO: implement compare_vocabulary_sizes()")


# ---------------------------------------------------------------------------
# Helper: Display vocabulary
# ---------------------------------------------------------------------------

def display_vocab(vocab: Dict[Tuple[str, ...], int], top_n: int = 20) -> None:
    """
    Print the top-n most frequent words and their current tokenization.

    Args:
        vocab:  BPE vocabulary dict.
        top_n:  Number of entries to display.
    """
    sorted_vocab = sorted(vocab.items(), key=lambda x: -x[1])
    print(f"\n{'Word representation':<40} {'Frequency'}")
    print("-" * 55)
    for word_tuple, freq in sorted_vocab[:top_n]:
        print(f"{' | '.join(word_tuple):<40} {freq}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Demonstrate BPE tokenization training and application.

    Steps:
        1. Train BPE with 20 merges; show vocabulary evolution.
        2. Tokenize sample words (including unseen morphological variants).
        3. Compare vocabulary sizes across different numbers of merges.
    """
    print("=" * 60)
    print("Byte Pair Encoding (BPE) Tokenization")
    print("=" * 60)

    # --- Initial vocabulary ---
    print("\n--- Initial character-level vocabulary ---")
    initial_vocab = get_vocab(SAMPLE_CORPUS)
    display_vocab(initial_vocab)

    # --- Train BPE ---
    print("\n--- Training BPE with 20 merges ---")
    final_vocab, merge_rules = train_bpe(SAMPLE_CORPUS, num_merges=20)

    print("\n--- Vocabulary after 20 merges ---")
    display_vocab(final_vocab)

    # --- Apply to new words ---
    print("\n--- Tokenizing words using learned merge rules ---")
    test_words = ["low", "lower", "lowest", "cool", "cooler", "newest", "fastest", "unknown"]
    for word in test_words:
        tokens = tokenize_bpe(word, merge_rules)
        print(f"  {word:<15} → {tokens}")

    # --- Vocabulary size comparison ---
    print("\n--- Vocabulary size vs number of merges ---")
    results = compare_vocabulary_sizes(SAMPLE_CORPUS, num_merges_list=[0, 5, 10, 20, 50])
    print(f"\n{'Merges':<10} {'Vocab size'}")
    print("-" * 22)
    for n_merges, vocab_size in sorted(results.items()):
        print(f"{n_merges:<10} {vocab_size}")

    # --- Show all merge rules ---
    print("\n--- Learned merge rules (first 15) ---")
    for i, rule in enumerate(merge_rules[:15]):
        print(f"  Rule {i+1:2d}: {rule[0]!r} + {rule[1]!r} → {rule[0]+rule[1]!r}")


if __name__ == "__main__":
    main()
