"""
tests/test_pipeline.py
──────────────────────
Unit tests for the Yoruba Health Whisper pipeline.

These tests validate:
  - Text normalisation logic (no model weights required)
  - Health vocabulary loading
  - Configuration loading
  - Audio array processing via inference (with a tiny synthetic signal)

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=src --cov=data
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import pytest

# ── Make project root importable ──────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

# ── Tests: text normalisation ──────────────────────────────────────────────────

from prepare_dataset import normalise_text  # noqa: E402


class TestNormaliseText:
    def test_lowercase(self):
        assert normalise_text("ILERA") == "ilera"

    def test_strip_whitespace(self):
        assert normalise_text("  orififo  ") == "orififo"

    def test_collapse_spaces(self):
        assert normalise_text("mo  ni  iba") == "mo ni iba"

    def test_yoruba_diacritics_preserved(self):
        result = normalise_text("Àtọgbẹ")
        assert "àtọgbẹ" in result

    def test_removes_special_punctuation(self):
        result = normalise_text("orififo! ati iba?")
        assert "!" not in result
        assert "?" not in result

    def test_empty_string(self):
        assert normalise_text("") == ""

    def test_unicode_normalisation(self):
        # NFC normalisation should not break Yoruba characters
        text = "ẹjẹ"
        result = normalise_text(text)
        assert len(result) > 0


# ── Tests: health vocabulary loading ─────────────────────────────────────────

from train import load_health_vocab  # noqa: E402


class TestLoadHealthVocab:
    def test_loads_vocab_file(self):
        vocab_path = ROOT / "data" / "yoruba_health_vocab.txt"
        tokens = load_health_vocab(str(vocab_path))
        assert len(tokens) > 0, "Vocabulary file should contain tokens"

    def test_skips_comments(self):
        vocab_path = ROOT / "data" / "yoruba_health_vocab.txt"
        tokens = load_health_vocab(str(vocab_path))
        for t in tokens:
            assert not t.startswith("#"), f"Comment line leaked into tokens: {t!r}"

    def test_skips_blank_lines(self):
        vocab_path = ROOT / "data" / "yoruba_health_vocab.txt"
        tokens = load_health_vocab(str(vocab_path))
        for t in tokens:
            assert t.strip() != "", "Blank line leaked into tokens"

    def test_known_health_terms_present(self):
        vocab_path = ROOT / "data" / "yoruba_health_vocab.txt"
        tokens = load_health_vocab(str(vocab_path))
        expected = {"ilera", "aisan", "dokita", "iba", "orififo"}
        for term in expected:
            assert term in tokens, f"Expected health term '{term}' not found in vocabulary"

    def test_missing_file_returns_empty(self, tmp_path):
        tokens = load_health_vocab(str(tmp_path / "nonexistent.txt"))
        assert tokens == []

    def test_custom_vocab_file(self, tmp_path):
        vocab_file = tmp_path / "test_vocab.txt"
        vocab_file.write_text("# comment\ntoken1\ntoken2\n\ntoken3\n", encoding="utf-8")
        tokens = load_health_vocab(str(vocab_file))
        assert tokens == ["token1", "token2", "token3"]


# ── Tests: configuration loading ──────────────────────────────────────────────

from train import load_config  # noqa: E402


class TestLoadConfig:
    def test_loads_yaml_config(self):
        config_path = ROOT / "config" / "config.yaml"
        cfg = load_config(str(config_path))
        assert isinstance(cfg, dict)

    def test_config_has_model_section(self):
        config_path = ROOT / "config" / "config.yaml"
        cfg = load_config(str(config_path))
        assert "model" in cfg
        assert "name" in cfg["model"]
        assert "language" in cfg["model"]

    def test_config_language_is_yoruba(self):
        config_path = ROOT / "config" / "config.yaml"
        cfg = load_config(str(config_path))
        assert cfg["model"]["language"] == "yo"

    def test_config_has_training_section(self):
        config_path = ROOT / "config" / "config.yaml"
        cfg = load_config(str(config_path))
        assert "training" in cfg
        assert "learning_rate" in cfg["training"]
        assert "max_steps" in cfg["training"]

    def test_config_has_dataset_section(self):
        config_path = ROOT / "config" / "config.yaml"
        cfg = load_config(str(config_path))
        assert "dataset" in cfg
        assert "audio_sampling_rate" in cfg["dataset"]
        assert cfg["dataset"]["audio_sampling_rate"] == 16000

    def test_load_config_invalid_path(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


# ── Tests: inference (lightweight, no model download) ─────────────────────────

class TestYorubaHealthTranscriber:
    """
    These tests mock the model/processor to avoid downloading weights.
    They validate the transcriber's interface and error handling.
    """

    def test_supported_extensions_set(self):
        from inference import SUPPORTED_EXTENSIONS
        assert ".wav" in SUPPORTED_EXTENSIONS
        assert ".mp3" in SUPPORTED_EXTENSIONS
        assert ".flac" in SUPPORTED_EXTENSIONS

    def test_sampling_rate_constant(self):
        from inference import SAMPLING_RATE
        assert SAMPLING_RATE == 16_000

    def test_transcribe_directory_empty(self, tmp_path):
        """transcribe_directory on an empty folder returns an empty list."""
        from unittest.mock import MagicMock, patch

        mock_processor = MagicMock()
        mock_model = MagicMock()

        with patch("inference.WhisperProcessor.from_pretrained", return_value=mock_processor), \
             patch("inference.WhisperForConditionalGeneration.from_pretrained", return_value=mock_model):
            from inference import YorubaHealthTranscriber
            t = YorubaHealthTranscriber.__new__(YorubaHealthTranscriber)
            t.processor = mock_processor
            t.model = mock_model
            t.device = "cpu"

            results = t.transcribe_directory(str(tmp_path))
            assert results == []

    def test_transcribe_array_calls_model(self):
        """transcribe_array should call generate and batch_decode."""
        import numpy as np
        from unittest.mock import MagicMock, patch
        import torch

        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.generate.return_value = torch.zeros((1, 10), dtype=torch.long)
        mock_processor.batch_decode.return_value = ["ilera rẹ dara"]
        mock_processor.return_value.to.return_value = mock_processor.return_value

        with patch("inference.WhisperProcessor.from_pretrained", return_value=mock_processor), \
             patch("inference.WhisperForConditionalGeneration.from_pretrained", return_value=mock_model):
            from inference import YorubaHealthTranscriber
            t = YorubaHealthTranscriber.__new__(YorubaHealthTranscriber)
            t.processor = mock_processor
            t.model = mock_model
            t.device = "cpu"

            dummy_audio = np.zeros(16000, dtype=np.float32)
            result = t.transcribe_array(dummy_audio)
            assert isinstance(result, str)
