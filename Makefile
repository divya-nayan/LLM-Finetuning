.PHONY: help install dev test lint format clean docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  install      Install dependencies"
	@echo "  dev          Install development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linters"
	@echo "  format       Format code"
	@echo "  clean        Clean cache and build files"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

dev:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e ".[dev,notebook]"
	pre-commit install

test:
	pytest tests -v --cov=src --cov-report=term-missing

lint:
	flake8 src tests
	isort --check-only src tests
	black --check src tests
	mypy src

format:
	isort src tests
	black src tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf outputs wandb

docker-build:
	docker build -t llm-finetuning:latest .

docker-run:
	docker run --gpus all -v $(PWD)/data:/app/data -v $(PWD)/outputs:/app/outputs llm-finetuning:latest

train:
	python -m src.training.train

evaluate:
	python -m src.evaluation.evaluate

serve:
	python -m src.inference serve outputs/final_model --port 8000