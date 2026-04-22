"""
data/prepare_dataset.py
───────────────────────
Download and preprocess Yoruba speech datasets for Whisper fine-tuning.

Supported sources
─────────────────
1. Mozilla Common Voice 11  (mozilla-foundation/common_voice_11_0, config "yo")
2. Google FLEURS            (google/fleurs, config "yo_ng")
3. Custom audio directory   (--custom_dir)

Usage
─────
# Common Voice (requires HF token for gated dataset)
python data/prepare_dataset.py --source common_voice --output_dir data/processed

# FLEURS (public)
python data/prepare_dataset.py --source fleurs --output_dir data/processed

# Custom WAV files  (directory must contain audio/ subdir and metadata.csv)
python data/prepare_dataset.py --source custom --custom_dir data/raw --output_dir data/processed
"""

from __future__ import annotations

import argparse
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

import datasets
from datasets import Audio, DatasetDict, load_dataset


# ── Constants ────────────────────────────────────────────────────────────────

SAMPLING_RATE = 16_000  # Whisper expects 16 kHz mono

COMMON_VOICE_DATASET = "mozilla-foundation/common_voice_11_0"
COMMON_VOICE_CONFIG = "yo"

FLEURS_DATASET = "google/fleurs"
FLEURS_CONFIG = "yo_ng"


# ── Text normalisation ───────────────────────────────────────────────────────

def normalise_text(text: str) -> str:
    """Lowercase, strip extra whitespace, remove non-Yoruba punctuation."""
    # Normalise unicode (NFC keeps Yoruba diacritics intact)
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    # Keep only letters (including accented Yoruba chars), spaces and apostrophes
    text = re.sub(r"[^\w\s'àáâãäèéêëìíîïòóôõöùúûüýÀÁÂÃÄÈÉÊËÌÍÎÏÒÓÔÕÖÙÚÛÜẹẸọỌṣṢ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Feature extraction helpers ───────────────────────────────────────────────

def prepare_common_voice(output_dir: Path, hf_token: Optional[str] = None) -> DatasetDict:
    """Download and preprocess Mozilla Common Voice Yoruba."""
    print("Loading Mozilla Common Voice 11 – Yoruba …")
    raw = load_dataset(
        COMMON_VOICE_DATASET,
        COMMON_VOICE_CONFIG,
        token=hf_token,
        trust_remote_code=True,
    )

    # Keep only necessary columns
    cols_to_remove = [
        c for c in raw["train"].column_names
        if c not in {"audio", "sentence"}
    ]
    raw = raw.remove_columns(cols_to_remove)

    # Resample audio to 16 kHz
    raw = raw.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    # Normalise transcripts
    def _normalise(batch):
        batch["sentence"] = normalise_text(batch["sentence"])
        return batch

    raw = raw.map(_normalise, desc="Normalising transcripts")

    # Drop empty transcripts
    raw = raw.filter(lambda x: len(x["sentence"]) > 0, desc="Removing empty transcripts")

    _save(raw, output_dir, prefix="common_voice")
    return raw


def prepare_fleurs(output_dir: Path, hf_token: Optional[str] = None) -> DatasetDict:
    """Download and preprocess FLEURS Yoruba."""
    print("Loading Google FLEURS – Yoruba …")
    raw = load_dataset(
        FLEURS_DATASET,
        FLEURS_CONFIG,
        token=hf_token,
        trust_remote_code=True,
    )

    cols_to_remove = [
        c for c in raw["train"].column_names
        if c not in {"audio", "transcription"}
    ]
    raw = raw.remove_columns(cols_to_remove)
    raw = raw.rename_column("transcription", "sentence")
    raw = raw.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    def _normalise(batch):
        batch["sentence"] = normalise_text(batch["sentence"])
        return batch

    raw = raw.map(_normalise, desc="Normalising transcripts")
    raw = raw.filter(lambda x: len(x["sentence"]) > 0, desc="Removing empty transcripts")

    _save(raw, output_dir, prefix="fleurs")
    return raw


def prepare_custom(custom_dir: Path, output_dir: Path) -> DatasetDict:
    """
    Load a custom dataset from a local directory.

    Expected layout
    ───────────────
    custom_dir/
      audio/          ← WAV files (16 kHz mono preferred)
      metadata.csv    ← columns: file_name, sentence

    metadata.csv example
    ─────────────────────
    file_name,sentence
    audio/001.wav,Mo ni orififo ati iba
    audio/002.wav,Dokita sọ pe mo ni malaria
    """
    metadata_path = custom_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.csv not found in {custom_dir}. "
            "Please create it with columns: file_name, sentence"
        )

    print(f"Loading custom dataset from {custom_dir} …")
    raw = load_dataset(
        "audiofolder",
        data_dir=str(custom_dir),
        trust_remote_code=True,
    )

    raw = raw.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    def _normalise(batch):
        batch["sentence"] = normalise_text(batch.get("sentence", batch.get("transcription", "")))
        return batch

    raw = raw.map(_normalise, desc="Normalising transcripts")
    raw = raw.filter(lambda x: len(x["sentence"]) > 0, desc="Removing empty transcripts")

    _save(raw, output_dir, prefix="custom")
    return raw


# ── Utility ──────────────────────────────────────────────────────────────────

def _save(dataset: DatasetDict, output_dir: Path, prefix: str) -> None:
    """Save a preprocessed DatasetDict to disk."""
    save_path = output_dir / prefix
    save_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(save_path))
    print(f"✓ Saved to {save_path}")
    for split, ds in dataset.items():
        print(f"  {split}: {len(ds):,} samples")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Yoruba health datasets for Whisper fine-tuning"
    )
    parser.add_argument(
        "--source",
        choices=["common_voice", "fleurs", "custom", "all"],
        default="fleurs",
        help="Dataset source to download (default: fleurs)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save processed datasets",
    )
    parser.add_argument(
        "--custom_dir",
        type=str,
        default=None,
        help="Path to custom audio directory (required when --source custom)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token (or set HF_TOKEN env var)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source in ("fleurs", "all"):
        prepare_fleurs(output_dir, hf_token=args.hf_token)

    if args.source in ("common_voice", "all"):
        prepare_common_voice(output_dir, hf_token=args.hf_token)

    if args.source in ("custom", "all"):
        if args.custom_dir is None:
            raise ValueError("--custom_dir is required when --source is 'custom' or 'all'")
        prepare_custom(Path(args.custom_dir), output_dir)

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()
