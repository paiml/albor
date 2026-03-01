# Appendix E: Leaderboard Strategy

### E.1 Target: Big Code Models Leaderboard

**URL**: https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard

The Big Code Models Leaderboard is the standard HuggingFace scoreboard for code
generation models. It evaluates HumanEval (Python pass@1) and MultiPL-E
(18 languages) with throughput measurements. ~60 models currently listed.

**Why this leaderboard**:
- Code generation focus — matches Albor's use case exactly
- HumanEval is our primary benchmark
- Accepts community submissions via PR
- **No sub-1B model has ever appeared** — Albor would be the first

**Current smallest entries (1B tier)**:

| Model | Params | HumanEval pass@1 |
|-------|--------|-------------------|
| phi-1 | 1.3B | 50.6% |
| DeciCoder-1B | 1.0B | 19.3% |
| SantaCoder | 1.1B | 18.1% |
| StarCoderBase-1B | 1.0B | 15.2% |

**Albor's position**: At >15% HumanEval with 350M params, Albor would be
competitive with the 1B tier at 1/3 the size. Even at >8% (base model), it
would establish the sub-1B category on the board.

**Submission process**:
1. Run `bigcode-evaluation-harness` (Python tool — the one exception to our
   zero-Python rule, because it is the leaderboard's own eval framework)
2. Standard params: top-p=0.95, temperature=0.2, n_samples=50,
   max_length_generation=512
3. Submit PR to `community_results/PAIML_ALBOR350M_noahgift/`
4. Include: scores JSON, generations folder, metrics folder
5. Results appear as "non-verified" (community submission)

### E.2 Why NOT Other Leaderboards

**Open LLM Leaderboard v2**: Benchmarks (IFEval, BBH, MATH L5, GPQA, MuSR,
MMLU-PRO) were designed for models >7B. A 350M model scores near random on
MATH Level 5 (~0%), GPQA (~25%), and MMLU-PRO (~10%). Waste of eval compute.

**EvalPlus Leaderboard**: Uses HumanEval+ and MBPP+ (80x more tests than
vanilla HumanEval). Secondary submission target if Big Code results are strong.
Currently no sub-1B models either. URL: https://evalplus.github.io/leaderboard.html

**BigCodeBench Leaderboard**: 1,140 software-engineering tasks. Designed for
7B+ models. A 350M model would score near zero. Not appropriate.

### E.3 General Capability Eval (Not a Leaderboard — Internal Only)

ARC-Easy, HellaSwag, PIQA, LAMBADA are the standard for sub-1B general model
comparison (Pythia, OPT, GPT-2 all publish on these). We evaluate on them for
internal comparison, but they have no dedicated leaderboard worth targeting.
Code benchmarks are the real scoreboard.

### E.4 FIM Evaluation

There is no canonical FIM benchmark. SantaCoder used a custom FIM evaluation;
other models use MultiPL-E or proprietary internal evals. Albor will define its
own FIM evaluation protocol (exact match on held-out Python functions) and
report absolute numbers rather than targeting a specific percentage.

### E.5 Falsification Risks for the Leaderboard Targets

1. **MoE→Dense distillation gap**: No published work demonstrates distilling
   an 80B MoE model into a 350M dense model. The architecture mismatch
   (DeltaNet+MoE routing → vanilla LLaMA) may limit knowledge transfer.
   If distillation gains are <2 points on HumanEval, the "Good" success
   criterion is at risk.

2. **Teacher inference bottleneck**: At ~2-5 tok/s (fp16 on Xeon), producing
   2B tokens of teacher logits takes ~12 days. If 500M tokens of logits
   proves insufficient, the timeline extends by weeks.

3. **Rust training stack maturity**: entrenar has never trained a model from
   scratch at 350M scale. Bugs in gradient accumulation, mixed precision,
   or checkpointing could cause silent correctness issues that only surface
   as poor benchmark scores.

4. **Data quality ceiling**: The local ground truth corpora (~71K files) are
   high quality but narrow. If the BPE tokenizer or data mix doesn't
   generalize well to HumanEval-style problems, the base model ceiling
   is lower than projected.

5. **bigcode-evaluation-harness compatibility**: The leaderboard eval tool is
   Python-based and expects HuggingFace-format models. Our SafeTensors export
   must be compatible with the harness's model loading. If not, we need a
   thin adapter — this is a potential gap not yet tracked.

### E.6 The Real Story

"A Python code completion model that was trained entirely in Rust with zero
Python dependencies — from data pipeline to on-device inference." The irony is
deliberate: a Rust ML stack producing a Python code assistant. The model is
the proof; the stack is the lasting value. Publishable
regardless of exact benchmark numbers.
