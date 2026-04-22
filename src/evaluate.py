"""
src/evaluate.py
───────────────
Evaluate a fine-tuned Whisper model on a Yoruba health dataset.

Reports:
  - Word Error Rate  (WER)
  - Character Error Rate (CER)

Usage
─────
python src/evaluate.py \
    --model_dir checkpoints/yoruba-health-whisper \
    --dataset_split test

To evaluate on custom audio + reference transcripts:
python src/evaluate.py \
    --model_dir checkpoints/yoruba-health-whisper \
    --dataset_split test \
    --dataset_path data/processed/custom
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import evaluate
import torch
from datasets import Audio, DatasetDict, load_from_disk
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLING_RATE = 16_000
BATCH_SIZE = 8


# ── Evaluation logic ──────────────────────────────────────────────────────────

def evaluate_model(
    model_dir: str,
    dataset_path: str,
    split: str,
    output_file: Optional[str] = None,
) -> dict:
    """
    Evaluate a Whisper checkpoint on a dataset split.

    Parameters
    ----------
    model_dir    : Path to the fine-tuned model directory.
    dataset_path : Path to the DatasetDict saved by prepare_dataset.py.
    split        : Dataset split to evaluate (train / validation / test).
    output_file  : Optional path to write per-sample predictions as JSONL.

    Returns
    -------
    dict with keys "wer" and "cer".
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading model from {model_dir} …")
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()

    print(f"Loading dataset from {dataset_path} (split={split}) …")
    dataset = load_from_disk(dataset_path)
    if isinstance(dataset, DatasetDict):
        ds = dataset[split]
    else:
        ds = dataset

    # Ensure audio is resampled
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    all_predictions: list[str] = []
    all_references: list[str] = []
    jsonl_rows: list[dict] = []

    for i in tqdm(range(0, len(ds), BATCH_SIZE), desc="Evaluating"):
        batch_samples = ds[i : i + BATCH_SIZE]
        audio_arrays = [a["array"] for a in batch_samples["audio"]]
        references = batch_samples["sentence"]

        inputs = processor(
            audio_arrays,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            pred_ids = model.generate(
                inputs.input_features,
                language="yo",
                task="transcribe",
            )

        predictions = processor.batch_decode(pred_ids, skip_special_tokens=True)

        all_predictions.extend(predictions)
        all_references.extend(references)

        if output_file:
            for ref, pred in zip(references, predictions):
                jsonl_rows.append({"reference": ref, "prediction": pred})

    wer = 100 * wer_metric.compute(predictions=all_predictions, references=all_references)
    cer = 100 * cer_metric.compute(predictions=all_predictions, references=all_references)

    print(f"\n{'─' * 40}")
    print(f"  WER : {wer:.2f}%")
    print(f"  CER : {cer:.2f}%")
    print(f"{'─' * 40}\n")

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            for row in jsonl_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Per-sample predictions saved to {out_path}")

    return {"wer": wer, "cer": cer}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Whisper model on Yoruba health data"
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--dataset_path",
        default="data/processed/fleurs",
        help="Path to the processed DatasetDict (default: data/processed/fleurs)",
    )
    parser.add_argument(
        "--dataset_split",
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Optional JSONL file to save per-sample predictions",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(
        model_dir=args.model_dir,
        dataset_path=args.dataset_path,
        split=args.dataset_split,
        output_file=args.output_file,
    )
