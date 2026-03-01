# Integration Tests

End-to-end pipeline tests that exercise the full training workflow.

## Purpose

Verify that all stack components (entrenar, alimentar, trueno, realizar, forjar)
work together correctly through the `apr` CLI.

## Tests

| Test | What It Checks | Expected Time |
|------|---------------|---------------|
| `data-pipeline` | Ingest -> quality check -> mix -> tokenize | ~30 minutes |
| `train-eval-loop` | Train 50M for 1000 steps, then eval | ~30 minutes |
| `distill-pipeline` | Pre-compute logits + distill (small subset) | ~1 hour |
| `post-training-ladder` | Merge -> prune -> quantize -> export (50M) | ~20 minutes |
| `pipeline-resume` | Interrupt and resume pipeline mid-execution | ~20 minutes |
| `cross-machine` | Dispatch tasks to both lambda and intel | ~30 minutes |

## Running

```bash
# Run all integration tests
apr test integration

# Run with the 50M model (faster)
apr test integration --model-size 50m

# Run specific test
apr test integration --filter train-eval-loop
```
