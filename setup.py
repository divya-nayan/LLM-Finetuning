"""Setup script for LLM Fine-tuning Framework."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="llm-finetuning",
    version="0.1.0",
    author="Divya Nayan",
    author_email="divyanayan88@gmail.com",
    description="A professional end-to-end LLM fine-tuning framework using PEFT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/divya-nayan/LLM-Finetuning",
    packages=find_packages(where="."),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "peft>=0.7.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.3",
        "scipy>=1.10.0",
        "wandb>=0.16.0",
        "tensorboard>=2.14.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "evaluate>=0.4.0",
        "rouge-score>=0.1.2",
        "nltk>=3.8.0",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.0",
        "rich>=13.5.0",
        "typer>=0.9.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
            "ipywidgets>=8.1.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "api": [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-finetune=src.training.train:main",
            "llm-inference=src.inference:main",
            "llm-evaluate=src.evaluation.evaluate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)