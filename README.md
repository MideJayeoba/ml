# Yoruba Health Speech Recognition with Whisper

A machine learning project for fine-tuning OpenAI's Whisper model to understand **Yoruba** language, specifically for health-related conversations and medical topics.

## Overview

This project fine-tunes [OpenAI Whisper](https://github.com/openai/whisper) using the [Hugging Face Transformers](https://huggingface.co/docs/transformers) library on Yoruba speech data with a focus on health and medical vocabulary. The goal is to build an accurate automatic speech recognition (ASR) system that can transcribe Yoruba health conversations.

### Why Yoruba + Health?

- Yoruba is spoken by over 45 million people, primarily in Nigeria and the diaspora.
- Medical information in indigenous languages dramatically improves healthcare access.
- Whisper's multilingual capability provides a strong base for fine-tuning on low-resource languages.

## Project Structure

```
ml/
├── config/
│   └── config.yaml          # Training hyperparameters and model settings
├── data/
│   ├── prepare_dataset.py   # Download and preprocess Yoruba speech datasets
│   └── yoruba_health_vocab.txt  # Health-specific Yoruba vocabulary
├── src/
│   ├── __init__.py
│   ├── train.py             # Fine-tuning script (Seq2Seq trainer)
│   ├── evaluate.py          # WER/CER evaluation script
│   └── inference.py         # Transcribe new Yoruba audio files
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py     # Unit tests for data prep and inference
├── requirements.txt
├── setup.py
└── README.md
```

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/MideJayeoba/ml.git
cd ml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare the dataset

The default dataset is [Mozilla Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) (Yoruba split) from Hugging Face. A Hugging Face account token is required for some datasets.

```bash
# Set your Hugging Face token (optional for public datasets)
export HF_TOKEN=your_token_here

# Download and preprocess Yoruba health data
python data/prepare_dataset.py --output_dir data/processed
```

You can also provide your own audio recordings. See `data/prepare_dataset.py` for the expected format.

### 3. Fine-tune Whisper

```bash
python src/train.py \
  --config config/config.yaml \
  --output_dir checkpoints/yoruba-health-whisper
```

Training progress is logged to the console and saved to `checkpoints/`.

### 4. Evaluate the model

```bash
python src/evaluate.py \
  --model_dir checkpoints/yoruba-health-whisper \
  --dataset_split test
```

This reports **Word Error Rate (WER)** and **Character Error Rate (CER)**.

### 5. Transcribe new audio

```bash
python src/inference.py \
  --model_dir checkpoints/yoruba-health-whisper \
  --audio path/to/yoruba_health_audio.wav
```

## Configuration

Edit `config/config.yaml` to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `openai/whisper-small` | Base Whisper model |
| `language` | `yo` | ISO 639-1 code for Yoruba |
| `max_steps` | `4000` | Training steps |
| `learning_rate` | `1e-5` | Learning rate |
| `per_device_train_batch_size` | `16` | Batch size per GPU |
| `gradient_accumulation_steps` | `2` | Gradient accumulation |

## Models

| Model | Parameters | VRAM | Notes |
|-------|-----------|------|-------|
| `whisper-tiny` | 39 M | ~1 GB | Fast, lower accuracy |
| `whisper-small` | 244 M | ~3 GB | **Recommended** |
| `whisper-medium` | 769 M | ~8 GB | Higher accuracy |
| `whisper-large-v3` | 1550 M | ~16 GB | Best accuracy |

## Dataset Sources

- **Mozilla Common Voice 11 – Yoruba**: ~30 hours, community-contributed
- **FLEURS – Yoruba** (`google/fleurs`, `yo_ng`): ~12 hours, high quality
- **Custom health recordings**: Place `.wav` files in `data/raw/` and run `prepare_dataset.py`

## Health Vocabulary

The file `data/yoruba_health_vocab.txt` contains curated Yoruba health terms (symptoms, conditions, greetings used in medical contexts). These tokens are added to the Whisper tokenizer during fine-tuning to improve recognition of domain-specific vocabulary.

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA GPU recommended (CPU training is very slow)

## License

MIT

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Mozilla Common Voice](https://commonvoice.mozilla.org/yo)