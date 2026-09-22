# Contributing to MZSAE

First off, thank you for considering contributing to MZSAE! It's people like you that make this project great.

## Prerequisites

To develop MZSAE, you will need:
* macOS (required for the Metal backend)
* Python 3.10 or higher
* Xcode Command Line Tools (`xcode-select --install`)
* PyTorch 2.0 or higher

*(Note: The CPU reference backend is available for all platforms, but Metal development requires Apple Silicon.)*

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mohamedhossammohamed/MZSAE.git
   cd MZSAE
   ```

2. **Compile Metal kernels:**
   ```bash
   make
   ```

3. **Install the package and dev dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

We use `ruff` to maintain code quality and formatting.
* Run the linter: `ruff check .`
* Run the formatter: `ruff format .`

Type hints are strongly encouraged for all new Python code to ensure maintainability and robustness.

## Testing

MZSAE relies on `pytest` for unit testing. Our suite includes 47 tests across 15 files.
* To run the test suite: `make test`
* All tests **must pass** before submitting a PR.
* If you are adding a new feature, please include corresponding unit tests.

## Pull Request Process

1. Fork the repository and create your branch from `main`.
2. Follow commit conventions (clear, descriptive commit messages).
3. Ensure your code passes all linting (`ruff check .`), formatting (`ruff format .`), and tests (`make test`).
4. Read and acknowledge the red-team audit in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) to understand the project's current boundaries and known issues.
5. Open a Pull Request on GitHub.

## Architecture Overview

For new contributors, here is a quick overview of the `src/mzsae` layout:
* **`MZSAEAttention`**: The core PyTorch module interface.
* **`MZSAEEngine`**: Handles the dual-plane KV cache logic and execution.
* **`Metal Kernels`**: Found in the corresponding `.metal` files, implementing MSL 3.1 features like GQA threadgroup sharing and size-aware buffer tracking.
* **`CPU Reference`**: Fallback implementations for non-macOS environments or testing.

## Benchmark Etiquette

When reporting performance improvements or benchmarks:
* **Be honest:** Do not cite per-layer performance as End-to-End (E2E) performance.
* **Be realistic:** Do not claim kernel math alone beats FlashAttention without full end-to-end context.
* **Contextualize:** Always include hardware info (e.g., Apple M1, M4 Max) and context length when posting numbers.
* Run our standard benchmark suites (e.g., `benchmarks/bench_selective_fetch.py`, real model weights benchmark with Qwen2.5-0.5B GGUF) for apples-to-apples comparisons.

Thank you for contributing!
