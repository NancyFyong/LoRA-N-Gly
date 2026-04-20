import argparse
import logging
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import EsmTokenizer

from model.esm_model import EsmModelClassification


SEED = 42


def setSeed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def readFastaRecords(filePath: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    currentHeader: Optional[str] = None
    currentSequenceParts: List[str] = []

    with open(filePath, "r") as fileHandle:
        for rawLine in fileHandle:
            line = rawLine.strip()
            if not line:
                continue
            if line.startswith(">"):
                if currentHeader is not None:
                    records.append((currentHeader, "".join(currentSequenceParts)))
                currentHeader = line[1:].strip() or f"record_{len(records) + 1}"
                currentSequenceParts = []
            else:
                currentSequenceParts.append(re.sub(r"[^a-zA-Z]", "", line).upper())

    if currentHeader is not None:
        records.append((currentHeader, "".join(currentSequenceParts)))

    return [(header, sequence) for header, sequence in records if sequence]


def parseInputSequences(args: argparse.Namespace) -> List[Tuple[str, str]]:
    if args.sequence and args.fasta_file:
        raise ValueError("Use either --sequence or --fasta_file, not both.")

    if args.sequence:
        sequence = re.sub(r"[^a-zA-Z]", "", args.sequence).upper()
        if not sequence:
            raise ValueError("--sequence is empty after cleaning.")
        return [(args.sequence_id, sequence)]

    if args.fasta_file:
        records = readFastaRecords(args.fasta_file)
        if not records:
            raise ValueError(f"No valid sequences found in FASTA: {args.fasta_file}")
        return records

    raise ValueError("No input provided. Use --sequence or --fasta_file.")


def resolveAttentionImplementation(requestedImpl: str, device: torch.device) -> str:
    if requestedImpl != "auto":
        return requestedImpl
    if device.type == "cuda":
        return "flash_attention_2"
    return "eager"


def loadModelAndTokenizer(args: argparse.Namespace, device: torch.device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    attentionImpl = resolveAttentionImplementation(args.attention_impl, device)
    candidateImpls = [attentionImpl]
    if attentionImpl == "flash_attention_2":
        candidateImpls.append("eager")

    baseModel = None
    lastError = None
    for impl in candidateImpls:
        logging.info("Loading base model %s with attn_implementation=%s", args.base_model, impl)
        try:
            baseModel = EsmModelClassification.from_pretrained(
                args.base_model,
                num_labels=2,
                torch_dtype=dtype,
                attn_implementation=impl,
            )
            attentionImpl = impl
            break
        except Exception as error:
            lastError = error
            logging.warning("attn_implementation=%s unavailable: %s", impl, error)

    if baseModel is None:
        raise RuntimeError(
            "Failed to load base model with requested attention implementations: "
            f"{candidateImpls}"
        ) from lastError

    logging.info("Loading LoRA adapter from %s", args.lora_model)
    model = PeftModel.from_pretrained(baseModel, args.lora_model)
    model = model.merge_and_unload()
    model.to(device)
    model.eval()

    tokenizer = EsmTokenizer.from_pretrained(args.base_model)
    logging.info("Model loaded. Effective attention implementation: %s", attentionImpl)
    return model, tokenizer


def inferResidueTokenOffset(tokenizer: EsmTokenizer) -> int:
    plainIds = tokenizer("A", add_special_tokens=False)["input_ids"]
    withSpecialIds = tokenizer("A", add_special_tokens=True)["input_ids"]
    targetId = plainIds[0]
    for tokenIdx, tokenId in enumerate(withSpecialIds):
        if tokenId == targetId:
            return tokenIdx
    return 1


def getCandidatePositions(sequence: str, glycoType: str) -> List[int]:
    candidateResidues = ["N"] if glycoType == "N" else ["S", "T"]
    return [idx + 1 for idx, aa in enumerate(sequence) if aa in candidateResidues]


def buildWindowGroups(
    sequenceLength: int,
    targetPositions1Based: List[int],
    windowSize: Optional[int],
) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    groupedByWindow: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)

    if windowSize is None or windowSize <= 0 or windowSize >= sequenceLength:
        fullWindow = (0, sequenceLength)
        for pos1Based in targetPositions1Based:
            groupedByWindow[fullWindow].append((pos1Based, pos1Based))
        return groupedByWindow

    halfWindow = windowSize // 2
    for pos1Based in targetPositions1Based:
        centerIdx = pos1Based - 1
        windowStart = max(0, centerIdx - halfWindow)
        windowEnd = min(sequenceLength, windowStart + windowSize)
        if (windowEnd - windowStart) < windowSize:
            windowStart = max(0, windowEnd - windowSize)
        localPos1Based = pos1Based - windowStart
        groupedByWindow[(windowStart, windowEnd)].append((pos1Based, localPos1Based))

    return groupedByWindow


def predictGivenPositions(
    model,
    tokenizer: EsmTokenizer,
    sequence: str,
    targetPositions1Based: List[int],
    batchSize: int,
    windowSize: Optional[int],
    device: torch.device,
    residueTokenOffset: int,
) -> Dict[int, Dict[str, float]]:
    sequenceLength = len(sequence)
    validPositions = sorted({pos for pos in targetPositions1Based if 1 <= pos <= sequenceLength})
    if not validPositions:
        return {}

    groupedByWindow = buildWindowGroups(sequenceLength, validPositions, windowSize)
    predictionMap: Dict[int, Dict[str, float]] = {}

    with torch.no_grad():
        for (windowStart, windowEnd), entries in sorted(groupedByWindow.items()):
            windowSequence = sequence[windowStart:windowEnd]
            encoded = tokenizer(windowSequence, return_tensors="pt")
            inputIds = encoded["input_ids"].to(device)
            attentionMask = encoded["attention_mask"].to(device)

            for startIdx in range(0, len(entries), batchSize):
                batchEntries = entries[startIdx : startIdx + batchSize]
                batchLen = len(batchEntries)

                batchInputIds = inputIds.repeat(batchLen, 1)
                batchAttentionMask = attentionMask.repeat(batchLen, 1)
                modelPos = [localPos - 1 + residueTokenOffset for _, localPos in batchEntries]
                posTensor = torch.tensor(modelPos, dtype=torch.long, device=device)

                outputs = model(
                    input_ids=batchInputIds,
                    attention_mask=batchAttentionMask,
                    pos=posTensor,
                )
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                predictedLabels = torch.argmax(logits, dim=-1).cpu().numpy()

                for idx, (globalPos, _) in enumerate(batchEntries):
                    predictionMap[globalPos] = {
                        "predicted_label": int(predictedLabels[idx]),
                        "prob_negative": float(probabilities[idx, 0]),
                        "prob_positive": float(probabilities[idx, 1]),
                    }

    return predictionMap


def predictSequenceCandidates(
    model,
    tokenizer: EsmTokenizer,
    sequence: str,
    sequenceId: str,
    glycoType: str,
    batchSize: int,
    windowSize: Optional[int],
    device: torch.device,
    residueTokenOffset: int,
) -> pd.DataFrame:
    candidatePositions = getCandidatePositions(sequence, glycoType)
    if not candidatePositions:
        return pd.DataFrame(
            columns=[
                "sequence_id",
                "position",
                "residue",
                "predicted_label",
                "prob_positive",
                "prob_negative",
            ]
        )

    predictionMap = predictGivenPositions(
        model=model,
        tokenizer=tokenizer,
        sequence=sequence,
        targetPositions1Based=candidatePositions,
        batchSize=batchSize,
        windowSize=windowSize,
        device=device,
        residueTokenOffset=residueTokenOffset,
    )

    rows = []
    for pos1Based in candidatePositions:
        predItem = predictionMap[pos1Based]
        rows.append(
            {
                "sequence_id": sequenceId,
                "position": pos1Based,
                "residue": sequence[pos1Based - 1],
                "predicted_label": predItem["predicted_label"],
                "prob_positive": predItem["prob_positive"],
                "prob_negative": predItem["prob_negative"],
            }
        )

    return pd.DataFrame(rows)


def printCandidateResults(resultDf: pd.DataFrame) -> None:
    if resultDf.empty:
        logging.info("No candidate residues found for the selected glycosylation type.")
        return

    for sequenceId, subDf in resultDf.groupby("sequence_id", sort=False):
        print(f"\n--- Sequence: {sequenceId} | candidate_sites={len(subDf)} ---")
        print("position\tresidue\tpredicted_label\tprob_positive\tprob_negative")
        for _, row in subDf.iterrows():
            print(
                f"{int(row['position'])}\t{row['residue']}\t{int(row['predicted_label'])}\t"
                f"{row['prob_positive']:.6f}\t{row['prob_negative']:.6f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict all candidate glycosylation sites for one sequence or FASTA."
    )
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--sequence_id", type=str, default="input_sequence")
    parser.add_argument("--fasta_file", type=str, default=None)
    parser.add_argument("--type", type=str, default="N", choices=["N", "O"])
    parser.add_argument("--base_model", type=str, default="facebook/esm2_t36_3B_UR50D")
    parser.add_argument("--lora_model", type=str, default="./lora_checkpoint")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Optional window size for long-sequence memory control. Default uses full sequence.",
    )
    parser.add_argument(
        "--attention_impl",
        type=str,
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--output_csv", type=str, default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    setSeed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    model, tokenizer = loadModelAndTokenizer(args, device)
    residueTokenOffset = inferResidueTokenOffset(tokenizer)

    inputRecords = parseInputSequences(args)
    allResults: List[pd.DataFrame] = []

    for sequenceId, sequence in inputRecords:
        resultDf = predictSequenceCandidates(
            model=model,
            tokenizer=tokenizer,
            sequence=sequence,
            sequenceId=sequenceId,
            glycoType=args.type,
            batchSize=args.batch_size,
            windowSize=args.window_size,
            device=device,
            residueTokenOffset=residueTokenOffset,
        )
        if not resultDf.empty:
            allResults.append(resultDf)

    finalDf = pd.concat(allResults, ignore_index=True) if allResults else pd.DataFrame()
    printCandidateResults(finalDf)

    if args.output_csv:
        finalDf.to_csv(args.output_csv, index=False)
        logging.info("Saved candidate predictions to %s", args.output_csv)


if __name__ == "__main__":
    main()
