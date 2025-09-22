#!/usr/bin/env python3
"""Script to prepare and validate training data."""

import json
import argparse
from pathlib import Path
import pandas as pd
from typing import List, Dict
from rich.console import Console
from rich.table import Table

console = Console()


def validate_instruction_format(data: List[Dict]) -> tuple:
    """Validate instruction-following dataset format.

    Args:
        data: List of data samples

    Returns:
        Tuple of (valid_samples, invalid_samples)
    """
    valid = []
    invalid = []

    required_fields = ["instruction", "output"]

    for idx, sample in enumerate(data):
        if all(field in sample for field in required_fields):
            valid.append(sample)
        else:
            invalid.append((idx, sample))

    return valid, invalid


def convert_to_instruction_format(
    input_file: str,
    output_file: str,
    input_format: str = "json"
):
    """Convert various formats to instruction format.

    Args:
        input_file: Input file path
        output_file: Output file path
        input_format: Input format (json, csv, jsonl)
    """
    input_path = Path(input_file)

    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        return

    # Load data based on format
    if input_format == "json":
        with open(input_path, "r") as f:
            data = json.load(f)
    elif input_format == "jsonl":
        data = []
        with open(input_path, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
    elif input_format == "csv":
        df = pd.read_csv(input_path)
        data = df.to_dict("records")
    else:
        console.print(f"[red]Unsupported format: {input_format}[/red]")
        return

    # Validate data
    valid, invalid = validate_instruction_format(data)

    # Display validation results
    table = Table(title="Data Validation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Samples", str(len(data)))
    table.add_row("Valid Samples", str(len(valid)))
    table.add_row("Invalid Samples", str(len(invalid)))

    console.print(table)

    if invalid:
        console.print("\n[yellow]Invalid samples (first 5):[/yellow]")
        for idx, sample in invalid[:5]:
            console.print(f"  Index {idx}: Missing fields")

    # Save valid data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(valid, f, indent=2)

    console.print(f"\n[green]Valid data saved to: {output_file}[/green]")


def split_dataset(
    input_file: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
):
    """Split dataset into train/val/test sets.

    Args:
        input_file: Input file path
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    """
    with open(input_file, "r") as f:
        data = json.load(f)

    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    # Save splits
    base_path = Path(input_file).parent

    with open(base_path / "train.json", "w") as f:
        json.dump(train_data, f, indent=2)

    with open(base_path / "eval.json", "w") as f:
        json.dump(val_data, f, indent=2)

    with open(base_path / "test.json", "w") as f:
        json.dump(test_data, f, indent=2)

    console.print(f"[green]Dataset split complete:[/green]")
    console.print(f"  Train: {len(train_data)} samples")
    console.print(f"  Val: {len(val_data)} samples")
    console.print(f"  Test: {len(test_data)} samples")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("input", help="Input file path")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--format", default="json", help="Input format")
    parser.add_argument("--split", action="store_true", help="Split dataset")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    args = parser.parse_args()

    if args.split:
        split_dataset(
            args.input,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio
        )
    else:
        output = args.output or "data/processed/data.json"
        convert_to_instruction_format(
            args.input,
            output,
            args.format
        )


if __name__ == "__main__":
    main()