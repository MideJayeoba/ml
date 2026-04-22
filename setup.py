from setuptools import setup, find_packages

setup(
    name="yoruba-health-whisper",
    version="0.1.0",
    description="Fine-tuning OpenAI Whisper for Yoruba health speech recognition",
    author="MideJayeoba",
    url="https://github.com/MideJayeoba/ml",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.39.0",
        "datasets>=2.18.0",
        "accelerate>=0.28.0",
        "evaluate>=0.4.1",
        "jiwer>=3.0.3",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "PyYAML>=6.0.1",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=5.0.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
