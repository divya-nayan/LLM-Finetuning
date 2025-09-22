# Contributing to LLM Fine-tuning Framework

Thank you for your interest in contributing to our LLM Fine-tuning Framework! We welcome contributions from the community.

## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)

## Code of Conduct

Please be respectful and inclusive in all interactions. We strive to maintain a welcoming environment for everyone.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/divya-nayan/LLM-Finetuning.git`
3. Add upstream remote: `git remote add upstream https://github.com/divya-nayan/LLM-Finetuning.git`

## How to Contribute

### Reporting Bugs
- Use the GitHub Issues tab
- Provide a clear description and steps to reproduce
- Include system information (OS, Python version, GPU)

### Suggesting Features
- Open a discussion in GitHub Discussions first
- Provide use cases and examples
- Consider implementation complexity

### Code Contributions
1. Check existing issues and PRs
2. Create an issue for discussion if needed
3. Fork and create a feature branch
4. Write code and tests
5. Submit a pull request

## Development Setup

```bash
# Install in development mode
pip install -e ".[dev,notebook]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run linting
black src tests
isort src tests
flake8 src tests
```

## Testing

All new features must include tests:

```python
# tests/test_your_feature.py
import unittest

class TestYourFeature(unittest.TestCase):
    def test_something(self):
        # Your test here
        pass
```

Run tests before submitting:
```bash
pytest tests/ --cov=src
```

## Pull Request Process

1. **Update your fork**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```

3. **Make changes and commit**
   ```bash
   git add .
   git commit -m "Add: your feature description"
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/my-feature
   ```

5. **PR Requirements**
   - Clear description of changes
   - Tests pass (automated CI)
   - Code follows style guide
   - Documentation updated if needed

## Style Guide

### Python Code Style
- Follow PEP 8
- Use Black for formatting (line length: 88)
- Use isort for imports
- Add type hints where possible

### Commit Messages
Format: `<type>: <description>`

Types:
- `Add:` New feature
- `Fix:` Bug fix
- `Update:` Enhancement
- `Refactor:` Code refactoring
- `Docs:` Documentation
- `Test:` Tests

Example:
```
Add: Support for Mixtral models
Fix: Memory leak in data loader
Update: Improve inference speed
```

### Documentation
- Use clear, concise language
- Include code examples
- Update README if adding features
- Add docstrings to functions:

```python
def my_function(param: str) -> str:
    """Brief description.

    Args:
        param: Parameter description

    Returns:
        Return value description
    """
    pass
```

## Project Structure

When adding new features, follow the existing structure:

```
src/
├── data/       # Data processing
├── models/     # Model-related code
├── training/   # Training logic
├── evaluation/ # Metrics and evaluation
├── utils/      # Shared utilities
```

## Questions?

- Open a GitHub Discussion
- Check existing issues
- Read the documentation

Thank you for contributing! 🚀