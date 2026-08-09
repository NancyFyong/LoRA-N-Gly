---
base_model: facebook/esm2_t30_150M_UR50D
library_name: peft
tags:
- biology
- protein-language-model
- glycosylation
---

# LoRA-N-Gly ESM2-150M Adapter

This directory contains the released LoRA adapter for N-linked glycosylation-site prediction with `facebook/esm2_t30_150M_UR50D`.

## Intended Use

The adapter is intended for sequence-based scoring of canonical N-linked candidate sequons. In the repository inference workflow, candidate positions are restricted to `N-X-[S/T]`, where `X` is not proline. Other Asn residues and non-candidate residues are not scored.

The model uses the full protein sequence as ESM-2 context and classifies the target candidate Asn residue embedding.

## Direct Inference

From the repository root:

```bash
python inference.py \
  --sequence_id example \
  --sequence MNNSTAAANVTNPS \
  --base_model facebook/esm2_t30_150M_UR50D \
  --lora_model checkpoints/N-linked/ESM-150M/checkpoint \
  --threshold 0.50
```

The output reports the 1-based candidate position, residue, sequon motif, predicted label, positive probability, negative probability, and threshold.

## Thresholds

- `0.05`: high-sensitivity screening.
- `0.50`: default benchmark-style decision threshold.
- `0.80`: high-precision conservative calling.

## Training Data and Task

The adapter was trained for binary classification of N-linked candidate sites using protein sequences and target candidate-site positions supplied in the repository CSV files. The LoRA target modules are `query`, `value`, and `out_proj`, with rank `r=8`, alpha `32`, and dropout `0.1`.

## Limitations

Predictions are sequence-based estimates and are not experimental validation. Negative examples may include unobserved or unannotated glycosylation sites. The released N-linked inference workflow does not score non-candidate residues.

## Framework Versions

The repository environment file pins compatible Python, PyTorch, Transformers, and PEFT versions for loading this adapter.
