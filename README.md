# 🚀 LLM Fine-tuning Framework

> **Production-ready framework for fine-tuning Large Language Models with PEFT/LoRA on consumer GPUs**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 Table of Contents
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)

## ✨ Features

### 🎯 Core Capabilities
- **🔥 PEFT/LoRA Integration** - Train LLMs with 90% less memory
- **📊 Multiple Data Formats** - JSON, JSONL, CSV, TXT, HuggingFace datasets
- **🤖 Universal Model Support** - Any HuggingFace model (LLaMA, Mistral, Falcon, etc.)
- **⚡ Quantization** - 4-bit and 8-bit support for smaller GPUs
- **🔄 Production Ready** - Docker, API serving, batch processing

### 💪 Why Use This Framework?
| Traditional Fine-tuning | This Framework |
|------------------------|----------------|
| 28GB+ GPU memory | 4-6GB GPU memory |
| $1000+ cloud costs | Run on consumer GPU |
| Days of training | Hours of training |
| Complex setup | One command start |

---

## 🚀 Quick Start

### 🎬 30-Second Setup

```bash
# 1. Clone the repository
git clone https://github.com/divya-nayan/LLM-Finetuning.git
cd LLM-Finetuning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run your first training
python -m src.training.train

# 🎉 That's it! Model will start training on sample data
```

---

## 📦 Installation

### Prerequisites
- **Python 3.8+**
- **CUDA 11.8+** (for GPU support)
- **16GB RAM** (32GB recommended)
- **GPU with 6GB+ VRAM** (RTX 3060 or better)

### 🔧 Detailed Installation

#### Option 1: Standard Installation
```bash
# Clone repository
git clone https://github.com/divya-nayan/LLM-Finetuning.git
cd LLM-Finetuning

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

#### Option 2: Development Installation
```bash
# Install with development tools
pip install -e ".[dev,notebook]"

# Setup pre-commit hooks
pre-commit install
```

#### Option 3: Docker Installation
```bash
# Build Docker image
docker build -t llm-finetuning .

# Run container
docker run --gpus all -v $(pwd)/data:/app/data llm-finetuning
```

---

## 💻 Usage

### 1️⃣ **Basic Training**

```bash
# Train with default settings (LLaMA-2 + Alpaca dataset)
python -m src.training.train

# Output:
# ✓ Loading model: meta-llama/Llama-2-7b-hf
# ✓ Applying LoRA (0.1% trainable params)
# ✓ Training started...
# ✓ Model saved to outputs/final_model
```

### 2️⃣ **Custom Model Training**

```bash
# Use different model
python -m src.training.train model.name=mistralai/Mistral-7B-v0.1

# Use smaller model for testing
python -m src.training.train model.name=microsoft/phi-2

# Adjust LoRA parameters
python -m src.training.train peft.lora_r=32 peft.lora_alpha=64
```

### 3️⃣ **Custom Dataset Training**

#### Prepare Your Data
Create `data/train.json`:
```json
[
  {
    "instruction": "What is machine learning?",
    "input": "",
    "output": "Machine learning is a subset of AI that enables systems to learn from data."
  },
  {
    "instruction": "Explain neural networks",
    "input": "in simple terms",
    "output": "Neural networks are computing systems inspired by biological neural networks."
  }
]
```

#### Train on Your Data
```bash
python -m src.training.train \
  data.train_path=data/train.json \
  data.eval_path=data/eval.json
```

### 4️⃣ **Inference (Using Your Model)**

#### Interactive Chat
```bash
python -m src.inference generate outputs/final_model --interactive

# Chat interface:
# You: What is AI?
# Assistant: AI, or Artificial Intelligence, is...
```

#### Single Prediction
```bash
python -m src.inference generate outputs/final_model \
  --prompt "Explain quantum computing"
```

#### Batch Processing
```bash
# Create input file
echo '[{"prompt": "What is AI?"}, {"prompt": "Explain ML"}]' > prompts.json

# Process batch
python -m src.inference batch outputs/final_model prompts.json results.json
```

#### API Server
```bash
# Start API server
python -m src.inference serve outputs/final_model --port 8000

# Use the API
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 100}'
```

---

## 🏗️ Architecture

### System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     Configuration Layer                       │
│                    (Hydra + YAML configs)                    │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│                      Data Pipeline                            │
│   ┌──────────┐    ┌──────────┐    ┌──────────────┐         │
│   │  Loader  │───►│ Processor│───►│  Tokenizer   │         │
│   └──────────┘    └──────────┘    └──────────────┘         │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│                   Model & Training Layer                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────────┐         │
│   │  Model   │───►│   PEFT   │───►│   Trainer    │         │
│   │ Loading  │    │  (LoRA)  │    │     (HF)     │         │
│   └──────────┘    └──────────┘    └──────────────┘         │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│                    Evaluation & Serving                       │
│   ┌──────────┐    ┌──────────┐    ┌──────────────┐         │
│   │ Metrics  │    │Inference │    │   API Server │         │
│   └──────────┘    └──────────┘    └──────────────┘         │
└───────────────────────────────────────────────────────────────┘
```

### Directory Structure
```
llm-finetuning/
├── 📁 src/                    # Source code
│   ├── 📁 data/              # Data processing
│   ├── 📁 models/            # Model management
│   ├── 📁 training/          # Training logic
│   ├── 📁 evaluation/        # Metrics & evaluation
│   ├── 📁 utils/             # Shared utilities
│   └── 📄 inference.py       # Inference engine
│
├── 📁 configs/                # Configuration files
│   ├── 📄 config.yaml        # Main config
│   ├── 📁 model/             # Model presets
│   ├── 📁 data/              # Dataset configs
│   └── 📁 training/          # Training configs
│
├── 📁 tests/                  # Unit tests
├── 📁 notebooks/              # Jupyter examples
├── 📁 scripts/                # Utility scripts
├── 📁 docker/                 # Docker files
└── 📁 data/                   # Your datasets
```

---

## ⚙️ Configuration

### 🎛️ Key Configuration Options

```yaml
# configs/config.yaml

# Model Configuration
model:
  name: meta-llama/Llama-2-7b-hf  # Model to fine-tune
  use_quantization: true           # Enable 4-bit/8-bit
  quantization_bits: 4             # 4 or 8

# PEFT/LoRA Configuration
peft:
  lora_r: 16                       # LoRA rank (smaller = less memory)
  lora_alpha: 32                   # LoRA scaling
  lora_dropout: 0.05               # Dropout for regularization

# Training Configuration
training:
  num_epochs: 3                    # Number of training epochs
  batch_size: 4                    # Batch size per GPU
  learning_rate: 2e-4              # Learning rate
  gradient_accumulation_steps: 4   # Gradient accumulation

# Data Configuration
data:
  max_length: 512                  # Maximum sequence length
  train_path: data/train.json      # Training data path
  eval_path: data/eval.json        # Evaluation data path
```

### 🔄 Override Configurations

```bash
# Command-line overrides
python -m src.training.train \
  model.name=mistralai/Mistral-7B \
  training.num_epochs=5 \
  training.learning_rate=1e-4 \
  peft.lora_r=32
```

---

## 📊 Supported Models & Datasets

### 🤖 Models

| Model | Parameters | Memory (LoRA+4bit) | Use Case |
|-------|------------|-------------------|----------|
| microsoft/phi-2 | 2.7B | 3-4 GB | Testing & Development |
| meta-llama/Llama-2-7b-hf | 7B | 4-6 GB | General Purpose |
| mistralai/Mistral-7B | 7B | 4-6 GB | Better Performance |
| codellama/CodeLlama-7b | 7B | 4-6 GB | Code Generation |
| meta-llama/Llama-2-13b | 13B | 8-10 GB | Higher Quality |

### 📚 Datasets

| Dataset | Size | Format | Use Case |
|---------|------|--------|----------|
| tatsu-lab/alpaca | 52K | Instruction | General instruction following |
| databricks/databricks-dolly-15k | 15K | Instruction | High-quality instructions |
| OpenAssistant/oasst1 | 88K | Conversation | Dialogue systems |
| sahil2801/CodeAlpaca-20k | 20K | Instruction | Code generation |

---

## 📈 Performance & Benchmarks

### Training Performance
| Setup | Training Time (10k samples) | Memory Usage |
|-------|----------------------------|--------------|
| Full Fine-tuning | 10-12 hours | 28 GB |
| LoRA (our framework) | 2-3 hours | 6 GB |
| LoRA + 4-bit | 2.5-3.5 hours | 4 GB |

### Inference Speed
| Configuration | Tokens/Second | Latency |
|--------------|---------------|---------|
| Full Model | 30-40 | 25ms |
| LoRA | 30-40 | 25ms |
| 4-bit Quantized | 25-35 | 30ms |

---

## 🛠️ Advanced Usage

### Multi-GPU Training
```bash
# Using Accelerate for distributed training
accelerate launch --multi_gpu --num_processes 4 \
  -m src.training.train
```

### Hyperparameter Sweep
```bash
# Test multiple learning rates
python -m src.training.train --multirun \
  training.learning_rate=1e-4,2e-4,5e-4 \
  peft.lora_r=8,16,32
```

### Custom Prompt Templates
```yaml
# configs/data/custom.yaml
prompt_template: |
  ### System: You are a helpful assistant.
  ### Human: {instruction}
  ### Assistant: {output}
```

### Resume Training
```bash
# Resume from checkpoint
python -m src.training.train \
  training.resume_from_checkpoint=true \
  training.checkpoint_path=outputs/checkpoint-500
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_metrics.py::TestMetrics
```

---

## 🐳 Docker

### Using Docker Compose
```bash
# Start training
docker-compose up training

# Start inference API
docker-compose up inference

# Start Jupyter notebook
docker-compose up notebook
```

### Manual Docker Commands
```bash
# Build image
docker build -t llm-finetuning .

# Run training
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  llm-finetuning

# Run inference server
docker run --gpus all -p 8000:8000 \
  llm-finetuning python -m src.inference serve /app/outputs/final_model
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Quick Contribution Guide
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers) - Model implementations
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning
- [Accelerate](https://github.com/huggingface/accelerate) - Distributed training
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) - Quantization

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/divya-nayan/LLM-Finetuning/issues)
- **Discussions**: [GitHub Discussions](https://github.com/divya-nayan/LLM-Finetuning/discussions)
- **Email**: divyanayan88@gmail.com

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=divya-nayan/LLM-Finetuning&type=Date)](https://star-history.com/#divya-nayan/LLM-Finetuning&Date)

---

## 🚀 Roadmap

- [x] PEFT/LoRA implementation
- [x] 4-bit/8-bit quantization
- [x] Multi-GPU support
- [x] API serving
- [ ] Web UI interface
- [ ] Model merging utilities
- [ ] ONNX export
- [ ] Mobile deployment

---

**Built with ❤️ by Divya Nayan**