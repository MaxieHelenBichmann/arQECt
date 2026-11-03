# Automated Hardware-Aware Analysis and Evaluation of Quantum Error-Correction Circuits

Sandbox repository for my Interdisciplinary Project in an Application Subject.

The "results" os this project will be contributed to [QECirc](https://github.com/qecirc/qecirc-website) and [MQT QECC](https://github.com/munich-quantum-toolkit/qecc).

# Idea

Give me a QEC circuit and a hardware platform (with some parameters), and I'll tell you how well suited this circuit is for the platform. "Well suited" here means a combination of compilation cost, runtime cost, and how the hardware-specific noise disrupts the circuit in its quality/effectiveness.

THIS IS NOT A COMPILER: I want to be as compiler-agnostic as possible (to stay applicable to many use cases), while still capturing the inherent properties and restrictions that the respective platforms carry. For this, I want to construct abstract hardware models that find a good balance between these aspects. 

## Scope

Supported Quantum Error-Correction Circuits:
- Encoding
- Logical-State Preparation
- Syndrome Extraction

Supported Hardware Platforms:
- Trapped Ions
- Neutral Atoms
- Superconducting Qubits


# Dev Setup


The project is managed with uv and requires Python 3.13 or newer. 

Create the virtual environment with the exact locked dependencies:

```bash
uv sync
```

Every command below runs inside this environment, either with `uv run <command>` or after `source .venv/bin/activate`.

Adding or removing dependencies (updates `pyproject.toml` and `uv.lock`):

```bash
uv add <package>
uv add --dev <package>
uv remove <package>
```

## Tests, Linter, Formatter, Type Checker

[![CI](https://github.com/MaxieHelenBichmann/arQECt/actions/workflows/ci.yml/badge.svg)](https://github.com/MaxieHelenBichmann/arQECt/actions/workflows/ci.yml)

```bash
uv run pytest                     # whole test suite in tests/
uv run pytest tests/models        # a single directory
uv run pytest -k dependency_dag   # tests matching a name
```

`ruff format` rewrites files in place, `ruff format --check` only reports:

```bash
uv run ruff check                 # lint (add --fix to apply safe fixes)
uv run ruff format                # format
uv run mypy src/                  # type-check
```

## Running Scripts

```bash
uv run python -m bm.model_accuracy.compiler
uv run python -m bm.performance.runner
```
