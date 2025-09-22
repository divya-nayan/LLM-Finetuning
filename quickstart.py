#!/usr/bin/env python3
"""
Quick Start Script for LLM Fine-tuning Framework

This script provides a simple interface to get started with the framework.
Run: python quickstart.py
"""

import os
import sys
import json
from pathlib import Path
import subprocess
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent))

def print_banner():
    """Print welcome banner."""
    print("\n" + "="*60)
    print("🚀 LLM Fine-tuning Framework - Quick Start")
    print("="*60 + "\n")

def check_dependencies():
    """Check if required dependencies are installed."""
    print("📦 Checking dependencies...")

    required = ["torch", "transformers", "peft", "accelerate"]
    missing = []

    for package in required:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        install = input("Install missing packages? (y/n): ")
        if install.lower() == 'y':
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing)
        else:
            print("Please install missing packages and run again.")
            sys.exit(1)

    print("✅ All dependencies installed!\n")

def select_mode():
    """Select operation mode."""
    print("🎯 Select mode:")
    print("  1. Quick Training (Small model, sample data)")
    print("  2. Custom Training (Choose model and data)")
    print("  3. Inference (Use existing model)")
    print("  4. Setup only (Install dependencies)")

    choice = input("\nEnter choice (1-4): ")
    return choice.strip()

def quick_training():
    """Run quick training with sample data."""
    print("\n🚀 Starting quick training...")
    print("  Model: microsoft/phi-2 (2.7B)")
    print("  Data: Example data (5 samples)")
    print("  Time: ~5-10 minutes on GPU\n")

    # Create sample data if not exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    if not (data_dir / "example_train.json").exists():
        print("📝 Creating sample training data...")
        sample_data = [
            {
                "instruction": "What is Python?",
                "input": "",
                "output": "Python is a high-level programming language."
            },
            {
                "instruction": "Explain machine learning",
                "input": "in simple terms",
                "output": "Machine learning is teaching computers to learn from data."
            }
        ]
        with open(data_dir / "example_train.json", "w") as f:
            json.dump(sample_data, f, indent=2)

    # Run training
    cmd = [
        sys.executable, "-m", "src.training.train",
        "model.name=microsoft/phi-2",
        "training.num_epochs=1",
        "training.logging_steps=1",
        "data.train_path=data/example_train.json",
        "data.eval_path=data/example_train.json"
    ]

    print("Running command:")
    print(" ".join(cmd))
    print("\n" + "-"*60)

    subprocess.run(cmd)

def custom_training():
    """Run custom training with user inputs."""
    print("\n🎯 Custom Training Setup")

    # Model selection
    print("\nAvailable models:")
    print("  1. meta-llama/Llama-2-7b-hf (7B, best quality)")
    print("  2. mistralai/Mistral-7B (7B, efficient)")
    print("  3. microsoft/phi-2 (2.7B, fastest)")
    print("  4. Custom model name")

    model_choice = input("\nSelect model (1-4): ")

    models = {
        "1": "meta-llama/Llama-2-7b-hf",
        "2": "mistralai/Mistral-7B-v0.1",
        "3": "microsoft/phi-2"
    }

    if model_choice in models:
        model_name = models[model_choice]
    else:
        model_name = input("Enter model name: ")

    # Data selection
    data_path = input("\nEnter training data path (or press Enter for example data): ")
    if not data_path:
        data_path = "data/example_train.json"

    # Training parameters
    epochs = input("Number of epochs (default: 3): ") or "3"
    batch_size = input("Batch size (default: 4): ") or "4"

    # Run training
    cmd = [
        sys.executable, "-m", "src.training.train",
        f"model.name={model_name}",
        f"training.num_epochs={epochs}",
        f"training.batch_size={batch_size}",
        f"data.train_path={data_path}",
        f"data.eval_path={data_path}"
    ]

    print("\n🚀 Starting training with:")
    print(f"  Model: {model_name}")
    print(f"  Data: {data_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print("\n" + "-"*60)

    subprocess.run(cmd)

def run_inference():
    """Run inference with existing model."""
    print("\n💬 Inference Mode")

    model_path = input("Enter model path (default: outputs/final_model): ")
    if not model_path:
        model_path = "outputs/final_model"

    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        print("Please train a model first or provide valid path.")
        return

    print("\nInference options:")
    print("  1. Interactive chat")
    print("  2. Single prompt")
    print("  3. Start API server")

    inf_choice = input("\nSelect option (1-3): ")

    if inf_choice == "1":
        cmd = [sys.executable, "-m", "src.inference", "generate", model_path, "--interactive"]
    elif inf_choice == "2":
        prompt = input("Enter prompt: ")
        cmd = [sys.executable, "-m", "src.inference", "generate", model_path, "--prompt", prompt]
    else:
        port = input("API port (default: 8000): ") or "8000"
        cmd = [sys.executable, "-m", "src.inference", "serve", model_path, "--port", port]
        print(f"\n🌐 API will be available at http://localhost:{port}")

    subprocess.run(cmd)

def setup_only():
    """Install dependencies only."""
    print("\n📦 Installing all dependencies...")

    cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    subprocess.run(cmd)

    print("\n✅ Setup complete! You can now:")
    print("  - Run training: python -m src.training.train")
    print("  - Run inference: python -m src.inference generate <model_path>")
    print("  - Check examples in notebooks/")

def main():
    """Main entry point."""
    print_banner()

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)

    # Check dependencies
    check_dependencies()

    # Select and run mode
    mode = select_mode()

    if mode == "1":
        quick_training()
    elif mode == "2":
        custom_training()
    elif mode == "3":
        run_inference()
    elif mode == "4":
        setup_only()
    else:
        print("Invalid choice")
        sys.exit(1)

    print("\n" + "="*60)
    print("✅ Complete! Check outputs/ directory for results.")
    print("📚 See README.md for more options.")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)