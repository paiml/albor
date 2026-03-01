# Smoke Tests

Quick sanity checks using the 50M model. These run in minutes, not hours.

## Purpose

Verify the entire stack works end-to-end before committing GPU days to the 350M run.

## Tests

| Test | What It Checks | Expected Time |
|------|---------------|---------------|
| `config-validation` | All YAML configs parse and pass `apr <cmd> plan` | <10 seconds |
| `train-50m-100steps` | Training loop converges for 100 steps | ~5 minutes |
| `checkpoint-save-load` | Save checkpoint, reload, verify weights match | ~1 minute |
| `tokenizer-roundtrip` | BPE encode/decode roundtrip on test data | <10 seconds |
| `pipeline-plan` | `apr pipeline plan` validates full DAG | <30 seconds |
| `contract-validate` | `pv validate contracts/*.yaml` | <10 seconds |

## Running

```bash
# Run all smoke tests
apr test smoke

# Run individual smoke test
apr test smoke --filter config-validation
```
