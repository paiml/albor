# bpe-tokenizer-kernel-v1

**Version:** 1.0.0

BPE tokenizer kernel — byte-pair encoding with lossless roundtrip

## References

- Sennrich et al. (2016) Neural Machine Translation of Rare Words with Subword Units
- Gage (1994) A New Algorithm for Data Compression

## Equations

### bpe_merge

$$
merge(a, b) = ab where (a,b) = argmin_{(p,q) in pairs} rank(p,q)
$$

**Domain:** $token sequence with adjacent pairs$

**Codomain:** $shorter token sequence$

**Invariants:**

- $Each merge reduces sequence length by at least 1$
- $Merge ordering is deterministic$
- $Final sequence uses only tokens in vocabulary$

### roundtrip

$$
decode(encode(x)) = x for all x in UTF-8
$$

**Domain:** $x: valid UTF-8 string$

**Codomain:** $encode(x): Vec<u32> where each id in [0, V)$

**Invariants:**

- $Lossless roundtrip for all valid UTF-8$
- $Empty input maps to empty output$
- $Byte-level fallback ensures all byte values representable$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Roundtrip lossless | $decode(encode(x)) = x for all valid UTF-8 x$ |
| 2 | invariant | Byte-level completeness | $Every byte value 0x00-0xFF is representable (no UNK)$ |
| 3 | idempotency | Deterministic encoding | $encode(x) = encode(x) for repeated calls on same input$ |
| 4 | invariant | Vocab size correctness | $len(tokenizer.vocab) = V (configured vocab size)$ |
| 5 | invariant | FIM sentinel tokens are atomic | $encode(<fim_prefix>) returns exactly one token ID$ |
| 6 | invariant | Empty input handling | $encode('') = [] and decode([]) = ''$ |

## Kernel Phases

1. **byte_encode**: Convert UTF-8 string to byte sequence — *bytes are valid UTF-8 representation*
2. **initial_tokenize**: Map bytes to initial token IDs (byte-level) — *all bytes have a token mapping*
3. **bpe_merge**: Iteratively apply BPE merge rules in priority order — *sequence length decreases monotonically*
4. **output**: Return final token ID sequence — *all IDs in [0, vocab_size)*

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-TOK-001 | Roundtrip invariant | decode(encode(x)) = x for random UTF-8 strings | Merge rule corrupts byte boundaries or special chars |
| FALSIFY-TOK-002 | Byte completeness | Every single-byte string encodes without UNK | Byte-level fallback tokens missing from vocabulary |
| FALSIFY-TOK-003 | Determinism | Same input always produces same tokens | Non-deterministic merge ordering (HashMap or thread race) |
| FALSIFY-TOK-004 | FIM sentinels | Each FIM sentinel token encodes to exactly one token | Sentinel tokens not added to vocabulary as special tokens |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-TOK-001 | TOK-INV-001 | 16 | exhaustive |

## QA Gate

**BPE Tokenizer Contract** (F-TOK-001)

Tokenizer correctness for Albor vocabulary

**Checks:** roundtrip_lossless, byte_completeness, deterministic_encoding, fim_sentinel_atomic

**Pass criteria:** All 4 falsification tests pass + Kani roundtrip harness verifies

