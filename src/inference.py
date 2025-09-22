"""Inference script for fine-tuned models."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel
import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt
import json
from src.utils.common import format_chat_prompt

app = typer.Typer()
console = Console()


class InferenceEngine:
    """Inference engine for fine-tuned models."""

    def __init__(
        self,
        model_path: str,
        base_model_path: Optional[str] = None,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """Initialize inference engine.

        Args:
            model_path: Path to fine-tuned model
            base_model_path: Path to base model (for PEFT models)
            device: Device to run on
            load_in_8bit: Load model in 8-bit
            load_in_4bit: Load model in 4-bit
        """
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

        self.model = None
        self.tokenizer = None
        self.streamer = None

        self.load_model()

    def load_model(self):
        """Load model and tokenizer."""
        console.print(f"[bold blue]Loading model from {self.model_path}...[/bold blue]")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": self.device,
            "trust_remote_code": True,
        }

        if self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        if self.base_model_path:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                **model_kwargs
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True)

        console.print("[bold green]Model loaded successfully![/bold green]")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        stream: bool = False,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Use sampling
            repetition_penalty: Repetition penalty
            stream: Stream output

        Returns:
            Generated text
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                streamer=self.streamer if stream else None,
            )

        output = self.tokenizer.decode(generated[0], skip_special_tokens=True)

        if not stream:
            response = output[len(prompt):].strip()
            return response

        return ""

    def chat(
        self,
        messages: list,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Chat interface for conversational models.

        Args:
            messages: List of chat messages
            max_new_tokens: Maximum new tokens
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Model response
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = format_chat_prompt(messages)

        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )



@app.command()
def generate(
    model_path: str = typer.Argument(..., help="Path to fine-tuned model"),
    prompt: str = typer.Option(None, "--prompt", "-p", help="Input prompt"),
    base_model: Optional[str] = typer.Option(None, "--base-model", "-b", help="Base model path for PEFT"),
    max_tokens: int = typer.Option(256, "--max-tokens", "-m", help="Maximum tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature"),
    top_p: float = typer.Option(0.95, "--top-p", help="Top-p sampling"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream output"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
):
    """Generate text using fine-tuned model."""
    engine = InferenceEngine(
        model_path=model_path,
        base_model_path=base_model,
    )

    if interactive:
        console.print("[bold cyan]Interactive mode. Type 'quit' to exit.[/bold cyan]")
        messages = []

        while True:
            user_input = Prompt.ask("\n[bold]You[/bold]")

            if user_input.lower() in ["quit", "exit"]:
                break

            messages.append({"role": "user", "content": user_input})

            console.print("\n[bold]Assistant[/bold]: ", end="")
            response = engine.chat(
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
            )

            if not stream:
                console.print(response)

            messages.append({"role": "assistant", "content": response})

    else:
        if not prompt:
            prompt = Prompt.ask("Enter prompt")

        console.print(f"\n[bold]Prompt:[/bold] {prompt}\n")
        console.print("[bold]Response:[/bold] ", end="")

        response = engine.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
        )

        if not stream:
            console.print(response)


@app.command()
def batch(
    model_path: str = typer.Argument(..., help="Path to fine-tuned model"),
    input_file: str = typer.Argument(..., help="Input JSON file with prompts"),
    output_file: str = typer.Argument(..., help="Output JSON file for responses"),
    base_model: Optional[str] = typer.Option(None, "--base-model", "-b", help="Base model path"),
    max_tokens: int = typer.Option(256, "--max-tokens", "-m", help="Maximum tokens"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Temperature"),
):
    """Batch inference on multiple prompts."""
    engine = InferenceEngine(
        model_path=model_path,
        base_model_path=base_model,
    )

    with open(input_file, "r") as f:
        prompts = json.load(f)

    results = []
    for item in prompts:
        if isinstance(item, str):
            prompt = item
        else:
            prompt = item.get("prompt", item.get("text", ""))

        console.print(f"Processing: {prompt[:50]}...")

        response = engine.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        results.append({
            "prompt": prompt,
            "response": response,
        })

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"[bold green]Saved {len(results)} responses to {output_file}[/bold green]")


@app.command()
def serve(
    model_path: str = typer.Argument(..., help="Path to fine-tuned model"),
    base_model: Optional[str] = typer.Option(None, "--base-model", "-b", help="Base model path"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind"),
):
    """Serve model as API endpoint."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        console.print("[bold red]FastAPI and uvicorn required for serving. Install with: pip install fastapi uvicorn[/bold red]")
        return

    app_api = FastAPI(title="LLM Inference API")

    class GenerateRequest(BaseModel):
        prompt: str
        max_tokens: int = 256
        temperature: float = 0.7
        top_p: float = 0.95

    class GenerateResponse(BaseModel):
        response: str

    engine = InferenceEngine(
        model_path=model_path,
        base_model_path=base_model,
    )

    @app_api.post("/generate", response_model=GenerateResponse)
    async def generate_endpoint(request: GenerateRequest):
        try:
            response = engine.generate(
                request.prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            return GenerateResponse(response=response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app_api.get("/health")
    async def health():
        return {"status": "healthy"}

    console.print(f"[bold green]Starting API server on {host}:{port}[/bold green]")
    uvicorn.run(app_api, host=host, port=port)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()