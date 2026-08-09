# LoRA-N-Gly

Sequence-based prediction of N-linked glycosylation sites by tuning ESM-2 protein language models with low-rank adaptation (LoRA).

![LoRA-N-Gly framework](./intro/framework.jpg "LoRA-N-Gly framework")

LoRA-N-Gly scores candidate N-linked glycosylation sequons from protein sequences. In N-linked mode, the inference code only scans canonical `N-X-[S/T]` sequons, where `X` is any amino acid except proline. Other Asn residues and all non-candidate residues are ignored. The full protein sequence is still used as ESM-2 context, and the classifier extracts the target candidate Asn embedding for binary prediction.

## Repository Contents

- `inference.py`: direct inference for one protein sequence or a FASTA file.
- `main.py`: training and benchmark-style evaluation with the provided CSV datasets.
- `model/esm_model.py`: ESM-2 sequence-classification wrapper that classifies the specified target residue position.
- `checkpoints/N-linked/ESM-3B/checkpoint`: released LoRA adapter for `facebook/esm2_t36_3B_UR50D`.
- `checkpoints/N-linked/ESM-150M/checkpoint`: released LoRA adapter for `facebook/esm2_t30_150M_UR50D`.
- `data/N-GlycositeAltas`: training, validation, and test CSV files used by the scripts.

## Environment

```bash
git clone <repository-url>
cd LoRA-N-Gly
conda env create -f env.yml
conda activate lora-n-gly
```

The ESM2-3B adapter requires substantial GPU memory. For a smaller inference run, use the ESM2-150M adapter and set `--base_model facebook/esm2_t30_150M_UR50D`.

## Direct Inference

Predict candidate N-linked sites for one protein sequence:

```bash
python inference.py \
  --sequence_id example \
  --sequence MNNSTAAANVTNPS \
  --base_model facebook/esm2_t36_3B_UR50D \
  --lora_model checkpoints/N-linked/ESM-3B/checkpoint \
  --threshold 0.50
```

Predict candidate N-linked sites for a FASTA file and save a CSV:

```bash
python inference.py \
  --fasta_file examples.fasta \
  --base_model facebook/esm2_t36_3B_UR50D \
  --lora_model checkpoints/N-linked/ESM-3B/checkpoint \
  --threshold 0.50 \
  --output_csv predictions.csv
```

Use the ESM2-150M adapter:

```bash
python inference.py \
  --sequence_id example \
  --sequence MNNSTAAANVTNPS \
  --base_model facebook/esm2_t30_150M_UR50D \
  --lora_model checkpoints/N-linked/ESM-150M/checkpoint \
  --threshold 0.50
```

Output columns:

- `sequence_id`: input sequence identifier.
- `position`: 1-based position of the candidate Asn residue.
- `residue`: target residue, `N` for N-linked prediction.
- `motif`: three-residue candidate sequon, such as `NVT`.
- `predicted_label`: `1` for predicted glycosylated, `0` for predicted non-glycosylated.
- `prob_positive`: predicted probability for the positive class.
- `prob_negative`: predicted probability for the negative class.
- `threshold`: probability threshold used to assign `predicted_label`.

## Thresholds

The default threshold is `0.50`, which is the standard binary decision threshold used for benchmark-style predictions. Depending on the use case, the inference script also supports alternative thresholds:

- `0.05`: high-sensitivity screening to retain more candidate sites.
- `0.50`: default balanced operating point.
- `0.80`: high-precision selection for more conservative positive calls.

The probability column is always reported, so users can apply a different threshold downstream without rerunning the model.

## Training and Evaluation

Train an ESM2-3B LoRA model with the provided N-linked data split:

```bash
bash scripts/train.sh
```

Evaluate the released ESM2-3B LoRA adapter on the configured test split:

```bash
bash scripts/predict.sh
```

The CSV files are expected to contain the protein sequence, the target candidate-site position, and the binary label. The training/evaluation code uses the full sequence as input and classifies only the supplied target candidate position.

## Biological Scope and Limitations

- The released inference workflow is intended for N-linked glycosylation candidate sites matching `N-X-[S/T]` with `X != P`.
- Non-candidate residues are not scored and are not interpreted as model negatives.
- Predictions estimate sequence-based glycosylation propensity and should not be interpreted as experimental validation.
- Negative labels in glycoproteomics-derived datasets may include unobserved or unannotated sites rather than experimentally proven non-glycosylated sites.
