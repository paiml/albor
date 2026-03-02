# Data Lineage

## Pipeline: Raw Code → Trained Model

```
Stage 1: Data Collection
  └── Python code corpus (100K lines, SHA-256 in PROVENANCE.md)

Stage 2: Tokenization
  └── alimentar tokenize → ByteLevel BPE v2 (32K vocab)
  └── Output: models/albor-tokenizer-v2/tokenizer.json

Stage 3: Pre-tokenization
  └── alimentar → Parquet format (seq_len=2048)
  └── Output: data/pretokenized-2048/{train,val}/

Stage 4: Training
  └── entrenar CudaTransformerTrainer (350M params)
  └── Config: configs/train/pretrain-350m.yaml
  └── Output: checkpoints/albor-base-350m/model.safetensors

Stage 5: Inference
  └── realizar run → Python code completion
  └── Input: prompt text
  └── Output: generated tokens
```

## Artifact Provenance

See `docs/PROVENANCE.md` for SHA-256 hashes of all artifacts.
