# Nebius Academy AI Performance Engineering: LLM Architecture

Coursework, notebooks, and implementation exercises for Module 2 of Nebius
Academy's [AI Performance Engineering](https://academy.nebius.com/ai-engineering-uk)
programme in London.

This repository is part of my applied ML engineering practice. It is intended to
show hands-on work with Python, PyTorch, transformer internals, local notebook
experimentation, and modern LLM fine-tuning workflows.

## Course Context

Nebius Academy's AI Performance Engineering programme is a practical 14-week
course for experienced developers. The wider programme covers LLM internals and
training, scalable deployment, MLOps, inference scaling, RAG systems, experiment
management, performance optimization, and post-training.

This repository focuses on Module 2: LLM Architecture. The module starts from ML
and neural-network foundations, then moves into sequence modelling, tokenization,
attention, transformer architectures, fine-tuning, LoRA, and practical LLM
inference concepts.

## Skills Practiced

- Python ML development with a reproducible virtual environment.
- PyTorch implementation of a small transformer language model.
- Embeddings, causal self-attention, multi-head attention, feed-forward blocks,
  residual connections, layer normalization, and language modelling loss.
- Local experimentation with Jupyter notebooks on CPU, CUDA, or Apple Silicon
  MPS.
- Parameter-efficient fine-tuning concepts using Hugging Face Transformers,
  PEFT, prompt tuning, and LoRA.
- Lightweight validation and project hygiene with `pytest`, `ruff`, and pinned
  dependencies.

## Structure

- `src/ml_course_practice/`: package code.
- `notebooks/week*/`: weekly hands-on notebooks.
- `notebooks/week05_06_transformers/practice/`: PEFT practice runbook and local helper code.
- `notebooks/week05_06_transformers/hometask/`: tiny transformer language-model homework runbook.
- `data/raw/`, `data/processed/`: local datasets/artifacts.
- `tests/`: automated tests.
- `docs/course_schedule.md`: course timeline and topics.

## Environment Setup

1. Create virtual environment:

```bash
python3 -m venv .venv
```

2. Activate:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

4. Optional Jupyter kernel registration:

```bash
python -m ipykernel install --user --name ml-course-practice --display-name "Python (ml-course-practice)"
```

## Run Notebooks Locally

Start Jupyter from the activated environment:

```bash
jupyter lab
```

Open a notebook and select the repo kernel:

```text
Kernel -> Change Kernel -> Python (ml-course-practice)
```

Check that the notebook is using the expected Python environment:

```python
import sys
import torch

print(sys.executable)
print(torch.__version__)
print("cuda:", torch.cuda.is_available())
print("mps:", torch.backends.mps.is_available())
```

On Apple laptops, the hometask notebook uses CUDA if available, then Apple MPS,
then CPU. Seeing `mps: True` is expected on supported Apple Silicon machines.

If a notebook raises `ModuleNotFoundError: No module named 'torch'`, the selected
Jupyter kernel is not the environment where dependencies were installed. Switch
to `Python (ml-course-practice)` or install dependencies into the selected kernel:

```python
import sys

!{sys.executable} -m pip install -r requirements-dev.txt
```

The hometask notebook is intentionally incomplete. `NotImplementedError` means
the current TODO section needs to be implemented before running the next check.

## Useful Commands

```bash
pytest
ruff check .
```
