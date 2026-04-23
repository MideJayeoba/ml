"""
src/train.py
────────────
Fine-tune OpenAI Whisper on Yoruba health speech data.

Usage
─────
python src/train.py --config config/config.yaml --output_dir checkpoints/yoruba-health-whisper

The script:
  1. Loads a pre-processed DatasetDict from disk (produced by data/prepare_dataset.py)
     or downloads the configured dataset on the fly.
  2. Adds health-specific Yoruba tokens to the tokenizer.
  3. Extracts log-Mel features using WhisperFeatureExtractor.
  4. Fine-tunes with Hugging Face Seq2SeqTrainer.
  5. Saves the best checkpoint to --output_dir.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
import yaml
from datasets import Audio, DatasetDict, load_dataset, load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: '{path}'. "
            "Please check the path or copy config/config.yaml to the expected location."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_health_vocab(vocab_path: str) -> List[str]:
    """Return non-empty, non-comment lines from the vocabulary file."""
    path = Path(vocab_path)
    if not path.exists():
        return []
    tokens = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tokens.append(line)
    return tokens


def load_dataset_from_config(cfg: Dict[str, Any], hf_token: Optional[str]) -> DatasetDict:
    """Load dataset from disk (if already prepared) or download on the fly."""
    processed_dir = Path("data/processed/fleurs")
    if processed_dir.exists():
        print(f"Loading pre-processed dataset from {processed_dir} …")
        return load_from_disk(str(processed_dir))

    print("Downloading FLEURS Yoruba dataset …")
    ds_cfg = cfg["dataset"]["secondary"]
    raw = load_dataset(ds_cfg["path"], ds_cfg["name"], token=hf_token, trust_remote_code=True)
    raw = raw.remove_columns(
        [c for c in raw["train"].column_names if c not in {"audio", "transcription"}]
    )
    raw = raw.rename_column("transcription", "sentence")
    raw = raw.cast_column("audio", Audio(sampling_rate=cfg["dataset"]["audio_sampling_rate"]))
    return raw


# ── Data collator ─────────────────────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Pad log-Mel features and token labels into fixed-size batches.
    Replaces padding token ids in labels with -100 so they are ignored by the loss.
    """

    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate inputs and labels before padding
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token prepended by tokenizer if present
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ── Preprocessing ──────────────────────────────────────────────────────────────

def build_prepare_fn(feature_extractor: WhisperFeatureExtractor, tokenizer: WhisperTokenizer):
    """Return a map function that converts raw audio samples to model inputs."""

    def prepare_dataset(batch):
        audio = batch["audio"]
        # Compute log-Mel spectrogram
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        # Tokenize target text
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    return prepare_dataset


# ── Metrics ────────────────────────────────────────────────────────────────────

def build_compute_metrics(tokenizer: WhisperTokenizer):
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id for decoding
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


# ── Main training loop ─────────────────────────────────────────────────────────

def train(config_path: str, output_dir: str, hf_token: Optional[str] = None) -> None:
    cfg = load_config(config_path)
    model_name: str = cfg["model"]["name"]
    language: str = cfg["model"]["language"]
    task: str = cfg["model"]["task"]
    tr_cfg = cfg["training"]

    print(f"Model  : {model_name}")
    print(f"Lang   : {language}  Task: {task}")

    # ── Tokenizer / Feature Extractor / Processor ──────────────────────────────
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)

    # Add domain-specific health vocabulary
    vocab_path = cfg["dataset"].get("health_vocab_path", "data/yoruba_health_vocab.txt")
    health_tokens = load_health_vocab(vocab_path)
    if health_tokens:
        num_added = tokenizer.add_tokens(health_tokens)
        print(f"Added {num_added} health-specific tokens to tokenizer.")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Resize embeddings if we added tokens
    if health_tokens:
        model.resize_token_embeddings(len(tokenizer))

    # Force language and task tokens during generation
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None

    # ── Dataset ────────────────────────────────────────────────────────────────
    dataset = load_dataset_from_config(cfg, hf_token)

    prepare_fn = build_prepare_fn(feature_extractor, tokenizer)

    print("Preprocessing dataset (this may take several minutes) …")
    dataset = dataset.map(
        prepare_fn,
        remove_columns=dataset["train"].column_names,
        num_proc=1,
        desc="Preparing features",
    )

    # ── Data Collator ─────────────────────────────────────────────────────────
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ── Training Arguments ────────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=tr_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tr_cfg["gradient_accumulation_steps"],
        learning_rate=tr_cfg["learning_rate"],
        warmup_steps=tr_cfg["warmup_steps"],
        max_steps=tr_cfg["max_steps"],
        weight_decay=tr_cfg.get("weight_decay", 0.01),
        fp16=tr_cfg.get("fp16", False) and torch.cuda.is_available(),
        bf16=tr_cfg.get("bf16", False),
        per_device_eval_batch_size=tr_cfg["per_device_eval_batch_size"],
        predict_with_generate=True,
        generation_max_length=tr_cfg.get("generation_max_length", 225),
        eval_strategy=tr_cfg.get("evaluation_strategy", "steps"),
        eval_steps=tr_cfg.get("eval_steps", 1000),
        save_steps=tr_cfg.get("save_steps", 1000),
        logging_steps=tr_cfg.get("logging_steps", 25),
        report_to=tr_cfg.get("report_to", "tensorboard"),
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=tr_cfg.get("push_to_hub", False),
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        processing_class=processor.feature_extractor,
    )

    print("Starting training …")
    trainer.train()

    # ── Save final model ──────────────────────────────────────────────────────
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"\nTraining complete. Model saved to: {output_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper on Yoruba health speech data"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override the output directory from config",
    )
    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token (or set HF_TOKEN env var)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg_raw = load_config(args.config)
    out_dir = args.output_dir or cfg_raw["training"]["output_dir"]
    train(args.config, out_dir, hf_token=args.hf_token)
