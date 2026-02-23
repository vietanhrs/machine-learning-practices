"""
Exercise 03: Beam Search Decoding
==================================
Implement beam search decoding for machine translation. Beam search maintains
multiple translation hypotheses at each step and returns the one with the
highest cumulative log-probability.

Learning Goals:
    - Understand why greedy decoding is suboptimal
    - Implement beam search with a priority queue (or sorted list)
    - Apply length normalization to counteract beam search's length bias
    - Compare beam search vs. greedy BLEU scores

Requirements:
    pip install torch numpy matplotlib
    (Assumes 02_seq2seq_attention.py has been completed — imports from it)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the toy task setup and model from the previous exercise
# (If running standalone, the key classes are reproduced below as stubs)
try:
    from exercises_02_seq2seq_attention import (
        Decoder, Encoder, Seq2SeqAttention,
        PAD, SOS, EOS,
        build_toy_dataset, build_vocab, encode_sentence,
        train_seq2seq, DIGIT_TO_WORD
    )
    _IMPORTED = True
except ImportError:
    _IMPORTED = False
    PAD, SOS, EOS = "<PAD>", "<SOS>", "<EOS>"
    DIGIT_TO_WORD = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
        5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    }


# ---------------------------------------------------------------------------
# Data class for a single beam hypothesis
# ---------------------------------------------------------------------------

@dataclass
class BeamHypothesis:
    """
    Represents a single hypothesis in the beam.

    Attributes:
        token_ids:   List of generated token indices so far (including SOS).
        hidden:      Current decoder hidden state tensor.
        log_prob:    Cumulative log-probability of this sequence.
        is_complete: True if EOS has been generated.
    """
    token_ids: List[int]
    hidden: torch.Tensor
    log_prob: float
    is_complete: bool = False

    def __lt__(self, other: "BeamHypothesis") -> bool:
        """Enable sorting by log_prob (higher is better)."""
        return self.log_prob > other.log_prob  # Reversed for max-heap behavior


# ---------------------------------------------------------------------------
# 1. Greedy Decode
# ---------------------------------------------------------------------------

def greedy_decode(
    model: "Seq2SeqAttention",
    src_tensor: torch.Tensor,
    tgt_vocab: Dict[str, int],
    tgt_idx_to_word: Dict[int, str],
    max_len: int = 30,
    device: Optional[torch.device] = None,
) -> List[str]:
    """
    Greedily decode a translation by always choosing the highest-probability token.

    Args:
        model:           Trained Seq2SeqAttention model.
        src_tensor:      LongTensor of shape (1, src_len) — single source sentence.
        tgt_vocab:       Target word → index mapping.
        tgt_idx_to_word: Target index → word mapping.
        max_len:         Maximum number of tokens to generate.
        device:          Torch device.

    Returns:
        List of predicted target tokens (excluding SOS/EOS/PAD).

    TODO:
        1. Set model to eval mode. Move src_tensor to device.
        2. Run encoder: encoder_outputs, hidden = model.encoder(src_tensor).
        3. Initialize dec_input = tensor([tgt_vocab[SOS]]) on device.
        4. Loop for up to max_len steps:
             a. logits, hidden, _ = model.decoder(dec_input, hidden, encoder_outputs)
             b. next_token = logits.argmax(dim=-1)  (greedy: highest prob)
             c. If next_token.item() == tgt_vocab[EOS]: break.
             d. Append next_token.item() to output list.
             e. Set dec_input = next_token.
        5. Convert output indices to words using tgt_idx_to_word.
        6. Return word list.
    """
    raise NotImplementedError("TODO: implement greedy_decode()")


# ---------------------------------------------------------------------------
# 2. Beam Search Decoder
# ---------------------------------------------------------------------------

class BeamSearchDecoder:
    """
    Beam search decoder for sequence generation.

    Maintains beam_size hypotheses at each step. At each decoding step,
    each hypothesis is expanded by all vocabulary tokens, producing
    beam_size * vocab_size candidates. The top beam_size are kept.

    Args:
        model:         Trained Seq2SeqAttention model.
        beam_size:     Number of hypotheses to maintain.
        max_len:       Maximum output sequence length.
        eos_token_id:  Index of the EOS token.
    """

    def __init__(
        self,
        model: "Seq2SeqAttention",
        beam_size: int = 4,
        max_len: int = 30,
        eos_token_id: int = 2,
    ) -> None:
        self.model = model
        self.beam_size = beam_size
        self.max_len = max_len
        self.eos_token_id = eos_token_id

    def decode(
        self,
        src_tensor: torch.Tensor,
        sos_token_id: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[List[int], float]:
        """
        Run beam search to find the best hypothesis.

        Args:
            src_tensor:   LongTensor of shape (1, src_len).
            sos_token_id: Index of the SOS token.
            device:       Torch device.

        Returns:
            best_token_ids: List of token indices for the best hypothesis
                            (excluding SOS and EOS).
            best_log_prob:  Normalized log-probability of the best hypothesis.

        TODO:
            1. Set model to eval mode.
            2. Run encoder: encoder_outputs, hidden = model.encoder(src_tensor).
               encoder_outputs: (1, src_len, hidden_dim)
               hidden: (1, 1, hidden_dim)

            3. Initialize beam with ONE hypothesis:
                   initial_hyp = BeamHypothesis(
                       token_ids=[sos_token_id],
                       hidden=hidden,
                       log_prob=0.0,
                   )
                   active_beams = [initial_hyp]

            4. completed_beams = []

            5. For each decoding step t from 1 to max_len:
                 candidates = []
                 For each hypothesis hyp in active_beams (that is not complete):
                     a. Get last token: dec_input = tensor([hyp.token_ids[-1]]).
                     b. Run decoder: logits, new_hidden, _ = model.decoder(dec_input, hyp.hidden, encoder_outputs).
                     c. Compute log-probs: log_probs = F.log_softmax(logits.squeeze(0), dim=-1).
                     d. Get top-k token candidates:
                            top_log_probs, top_ids = self._get_top_k_tokens(log_probs, self.beam_size)
                     e. For each (token_id, token_log_prob) in zip(top_ids, top_log_probs):
                            new_log_prob = hyp.log_prob + token_log_prob.item()
                            is_complete  = (token_id == self.eos_token_id)
                            new_hyp = BeamHypothesis(
                                token_ids=hyp.token_ids + [token_id],
                                hidden=new_hidden,
                                log_prob=new_log_prob,
                                is_complete=is_complete,
                            )
                            candidates.append(new_hyp)

                 f. Sort candidates by log_prob / length (normalized):
                        candidates.sort(
                            key=lambda h: self._normalize_scores(h.log_prob, len(h.token_ids))
                        )
                 g. Keep top beam_size:
                        active_beams = []
                        for hyp in candidates:
                            if hyp.is_complete:
                                completed_beams.append(hyp)
                            else:
                                if len(active_beams) < self.beam_size:
                                    active_beams.append(hyp)
                        If all beams complete or active_beams is empty: break.

            6. If no completed beams, use the best active beam.
               Otherwise, select the completed beam with the highest normalized score.
            7. Extract token_ids (remove SOS and EOS), return with best score.
        """
        raise NotImplementedError("TODO: implement BeamSearchDecoder.decode()")

    def _get_top_k_tokens(
        self,
        log_probs: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Get the top-k log-probabilities and their token indices.

        Args:
            log_probs: 1D tensor of log-probabilities over the vocabulary.
            k:         Number of top tokens to return.

        Returns:
            top_log_probs: Tensor of the k highest log-probabilities.
            top_ids:       List of k token indices corresponding to those probabilities.

        TODO:
            1. Use torch.topk(log_probs, k) to get top-k values and indices.
            2. Return (values, indices.tolist()).
        """
        raise NotImplementedError("TODO: implement BeamSearchDecoder._get_top_k_tokens()")

    def _normalize_scores(
        self,
        log_prob: float,
        length: int,
        length_penalty_alpha: float = 0.6,
    ) -> float:
        """
        Apply length normalization to prevent beam search from favoring short sequences.

        Length penalty (Wu et al., 2016 — Google NMT):
            lp(Y) = (5 + len(Y))^α / (5 + 1)^α

        Normalized score = log_prob / lp(length)

        Args:
            log_prob:             Cumulative log-probability.
            length:               Sequence length (number of tokens).
            length_penalty_alpha: Strength of the length penalty (0 = no penalty, 1 = strong).

        Returns:
            Normalized score (higher = better).

        TODO:
            1. Compute lp = ((5 + length) ** length_penalty_alpha) / ((5 + 1) ** length_penalty_alpha).
            2. Return log_prob / lp.

        Note:
            Returning a higher value means the hypothesis is ranked higher.
            This method is used as the key function for sorting.
        """
        raise NotImplementedError("TODO: implement BeamSearchDecoder._normalize_scores()")


# ---------------------------------------------------------------------------
# 3. Compare Beam Search vs. Greedy
# ---------------------------------------------------------------------------

def compare_beam_vs_greedy(
    model: "Seq2SeqAttention",
    test_sentences: List[List[str]],
    src_vocab: Dict[str, int],
    tgt_vocab: Dict[str, int],
    tgt_idx_to_word: Dict[int, str],
    beam_sizes: Optional[List[int]] = None,
    max_len: int = 20,
    device: Optional[torch.device] = None,
) -> None:
    """
    Compare translation quality of greedy decoding vs. beam search with different beam sizes.

    For each test sentence:
        1. Translate with greedy decoding.
        2. Translate with beam search (multiple beam sizes).
        3. Compute BLEU-4 for each.
        4. Print a comparison table.

    Args:
        model:           Trained Seq2SeqAttention model.
        test_sentences:  List of source token lists to translate.
        src_vocab:       Source vocabulary.
        tgt_vocab:       Target vocabulary.
        tgt_idx_to_word: Target index-to-word mapping.
        beam_sizes:      List of beam sizes to compare (default: [1, 2, 4, 8]).
        max_len:         Maximum generation length.
        device:          Torch device.

    TODO:
        1. For each sentence in test_sentences:
             a. Compute the expected (ground-truth) translation.
             b. Encode source sentence → src_tensor.
             c. Greedy decode: translation = greedy_decode(model, src_tensor, ...).
             d. For each beam_size in beam_sizes:
                    decoder = BeamSearchDecoder(model, beam_size, max_len, eos_id)
                    token_ids, score = decoder.decode(src_tensor, sos_id, device)
                    beam_translation = [tgt_idx_to_word[id] for id in token_ids]

        2. Compute sentence-level BLEU for each approach.
           (Import or re-implement bleu_score from exercise 01.)
        3. Print a formatted comparison table.
    """
    raise NotImplementedError("TODO: implement compare_beam_vs_greedy()")


# ---------------------------------------------------------------------------
# Helper: Simple sentence-level BLEU (abbreviated from exercise 01)
# ---------------------------------------------------------------------------

def _simple_bleu(hypothesis: List[str], reference: List[str], max_n: int = 4) -> float:
    """
    Compute a simplified sentence-level BLEU score.
    (Simplified — see Exercise 01 for the full implementation.)
    """
    from collections import Counter

    def ngrams(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    if len(hypothesis) == 0:
        return 0.0

    # Brevity penalty
    bp = 1.0 if len(hypothesis) >= len(reference) else math.exp(1 - len(reference) / max(len(hypothesis), 1))

    log_prec = 0.0
    for n in range(1, max_n + 1):
        hyp_counts = ngrams(hypothesis, n)
        ref_counts = ngrams(reference, n)
        clipped = sum(min(c, ref_counts.get(g, 0)) for g, c in hyp_counts.items())
        total = max(sum(hyp_counts.values()), 1)
        prec = clipped / total
        if prec == 0:
            return 0.0
        log_prec += (1.0 / max_n) * math.log(prec)

    return bp * math.exp(log_prec)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Demonstrate beam search on the number-to-word translation task.

    Steps:
        1. Build dataset and vocabularies.
        2. Train (or load) the Seq2Seq model.
        3. Compare greedy vs. beam search (beam sizes 1, 2, 4, 8).
        4. Show diversity of beam hypotheses for one sentence.
    """
    import random
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not _IMPORTED:
        print("\nNote: Could not import from 02_seq2seq_attention.py.")
        print("Please complete Exercise 02 first, then re-run this script.")
        print("Alternatively, copy the model classes and training code here.")
        return

    # --- Data ---
    MAX_LEN = 7
    src_sentences, tgt_sentences = build_toy_dataset(n_samples=2000, max_len=5)
    src_vocab, src_idx_to_word = build_vocab(src_sentences)
    tgt_vocab, tgt_idx_to_word = build_vocab(tgt_sentences)

    src_encoded = [encode_sentence(s, src_vocab, add_eos=True, max_len=MAX_LEN) for s in src_sentences]
    tgt_encoded = [encode_sentence(t, tgt_vocab, add_sos=True, add_eos=True, max_len=MAX_LEN+1) for t in tgt_sentences]

    src_tensor_all = torch.LongTensor(src_encoded)
    tgt_tensor_all = torch.LongTensor(tgt_encoded)

    # --- Model ---
    encoder = Encoder(len(src_vocab), embed_dim=32, hidden_dim=64)
    decoder = Decoder(len(tgt_vocab), embed_dim=32, hidden_dim=64)
    model   = Seq2SeqAttention(encoder, decoder)

    print("\nTraining model...")
    train_seq2seq(model, src_tensor_all, tgt_tensor_all,
                  n_epochs=40, lr=1e-3, batch_size=64, device=device)

    # --- Compare beam vs greedy ---
    test_cases = [
        ["1", "2", "3"],
        ["9", "0", "5", "3"],
        ["7", "7", "7"],
        ["4", "2", "8", "1"],
        ["0"],
    ]
    compare_beam_vs_greedy(
        model, test_cases, src_vocab, tgt_vocab, tgt_idx_to_word,
        beam_sizes=[1, 2, 4, 8], device=device
    )

    # --- Length penalty effect ---
    print("\n--- Length Penalty Sensitivity ---")
    decoder_bs = BeamSearchDecoder(model, beam_size=4, max_len=20, eos_token_id=tgt_vocab[EOS])
    src_example = ["3", "1", "4", "1", "5"]
    src_enc = encode_sentence(src_example, src_vocab, add_eos=True, max_len=MAX_LEN)
    src_t   = torch.LongTensor([src_enc]).to(device)
    expected = [DIGIT_TO_WORD[int(d)] for d in src_example]
    print(f"\nSource:   {' '.join(src_example)}")
    print(f"Expected: {' '.join(expected)}")

    for alpha in [0.0, 0.3, 0.6, 1.0]:
        decoder_bs.eos_token_id = tgt_vocab[EOS]
        # Temporarily patch the alpha
        original_normalize = decoder_bs._normalize_scores
        decoder_bs._normalize_scores = lambda lp, length, lpa=alpha: (
            lp / (((5 + length) ** lpa) / ((5 + 1) ** lpa))
        )
        ids, score = decoder_bs.decode(src_t, tgt_vocab[SOS], device=device)
        words = [tgt_idx_to_word.get(i, "?") for i in ids if i not in (tgt_vocab[PAD], tgt_vocab[EOS])]
        bleu = _simple_bleu(words, expected)
        print(f"  α={alpha:.1f}  Translation: {' '.join(words):<30}  BLEU: {bleu:.4f}")
        decoder_bs._normalize_scores = original_normalize


if __name__ == "__main__":
    main()
