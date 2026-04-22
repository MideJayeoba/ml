"""
src/inference.py
────────────────
Transcribe Yoruba health audio using a fine-tuned Whisper model.

Usage
─────
# Transcribe a single audio file
python src/inference.py \
    --model_dir checkpoints/yoruba-health-whisper \
    --audio path/to/yoruba_health_audio.wav

# Transcribe all WAV files in a directory
python src/inference.py \
    --model_dir checkpoints/yoruba-health-whisper \
    --audio_dir path/to/audio_folder \
    --output_file transcriptions.jsonl

# Use the base Whisper model (no fine-tuning required)
python src/inference.py \
    --model_dir openai/whisper-small \
    --audio path/to/audio.wav
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLING_RATE = 16_000
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


# ── Core transcription ─────────────────────────────────────────────────────────

class YorubaHealthTranscriber:
    """
    Load a fine-tuned (or base) Whisper model and transcribe Yoruba audio.
    """

    def __init__(self, model_dir: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from '{model_dir}' on {self.device} …")
        self.processor = WhisperProcessor.from_pretrained(model_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def transcribe_array(self, audio_array, sampling_rate: int = SAMPLING_RATE) -> str:
        """Transcribe a raw numpy audio array."""
        inputs = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(
                inputs.input_features,
                language="yo",
                task="transcribe",
            )

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()

    def transcribe_file(self, audio_path: str) -> str:
        """Load an audio file and return its Yoruba transcription."""
        import librosa

        audio_path = str(audio_path)
        array, sr = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
        return self.transcribe_array(array, sampling_rate=SAMPLING_RATE)

    def transcribe_directory(self, audio_dir: str) -> list[dict]:
        """
        Transcribe all supported audio files in a directory.

        Returns a list of dicts: [{"file": "...", "transcription": "..."}, ...]
        """
        audio_dir = Path(audio_dir)
        results = []
        audio_files = sorted(
            f for f in audio_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not audio_files:
            print(f"No supported audio files found in {audio_dir}")
            return results

        print(f"Transcribing {len(audio_files)} file(s) from {audio_dir} …\n")
        for path in audio_files:
            try:
                text = self.transcribe_file(str(path))
                results.append({"file": str(path), "transcription": text})
                print(f"[{path.name}]  {text}")
            except Exception as exc:
                print(f"[{path.name}]  ERROR: {exc}")
                results.append({"file": str(path), "transcription": None, "error": str(exc)})

        return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe Yoruba health audio with a fine-tuned Whisper model"
    )
    parser.add_argument(
        "--model_dir",
        default="checkpoints/yoruba-health-whisper",
        help="Path to fine-tuned model directory (or Hugging Face model ID)",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", help="Path to a single audio file")
    source.add_argument("--audio_dir", help="Directory containing audio files")

    parser.add_argument(
        "--output_file",
        default=None,
        help="Optional JSONL file to save transcriptions (for --audio_dir)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override: 'cpu', 'cuda', 'mps' (default: auto-detect)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transcriber = YorubaHealthTranscriber(model_dir=args.model_dir, device=args.device)

    if args.audio:
        text = transcriber.transcribe_file(args.audio)
        print(f"\nTranscription:\n  {text}\n")

    elif args.audio_dir:
        results = transcriber.transcribe_directory(args.audio_dir)
        if args.output_file:
            out = Path(args.output_file)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as fh:
                for row in results:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
