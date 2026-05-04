# Tiny Transformer LM Hometask

This folder contains the Week 5-6 transformer language-model hometask for the
Nebius Academy LLM Architecture module.

The main submission notebook is:

```text
tiny_transformer_lm_Alex_20260504_ver1.ipynb
```

The original working notebook is:

```text
tiny_transformer_lm.ipynb
```

## Task Summary

The notebook implements a small character-level causal Transformer language
model in PyTorch. The completed components are:

- `MultiHeadSelfAttention`
- `FeedForward`
- `Block`
- `TinyTransformerLM`
- autoregressive `generate`

The model is trained to predict the next character from a Shakespeare-style text
dataset.

## Implementation Notes

The attention module uses:

- a single Q/K/V projection with `nn.Linear(n_embd, 3 * n_embd, bias=False)`
- multi-head reshaping from `(B, T, C)` to `(B, n_head, T, head_size)`
- scaled dot-product attention
- a registered lower-triangular causal mask
- softmax over key positions
- attention dropout and residual dropout
- output projection back to `(B, T, C)`

The Transformer block uses pre-norm residual connections:

```python
x = x + self.attn(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

The language model combines token embeddings and positional embeddings, applies
a stack of Transformer blocks, then predicts next-character logits with an LM
head.

## Submission Checklist

Before submission, the notebook should be saved with all cells executed so the
outputs are visible to reviewers.

Required checks:

- all `raise NotImplementedError` lines removed from implemented TODOs
- attention sanity-check cell passes
- final train loss below `1.8`
- training curve visible
- generated sample visible
- final metrics table visible

Observed attention sanity-check output:

```text
Attention looks good: correct shape, causality preserved, variable-T OK
```

## Local MPS Run

The notebook was tested locally on Apple Silicon MPS.

Training log:

```text
iter     0 | train loss 4.2123 | val loss 4.2108
iter   500 | train loss 1.8082 | val loss 1.9239
iter  1000 | train loss 1.5655 | val loss 1.7460
iter  1500 | train loss 1.4499 | val loss 1.6538
iter  2000 | train loss 1.3851 | val loss 1.5989
iter  2500 | train loss 1.3348 | val loss 1.5566
iter  3000 | train loss 1.2997 | val loss 1.5454
iter  3500 | train loss 1.2729 | val loss 1.5220
iter  4000 | train loss 1.2441 | val loss 1.5155
iter  4500 | train loss 1.2246 | val loss 1.5080
iter  5000 | train loss 1.2065 | val loss 1.4946
```

Final local metric summary:

```text
                         nll (nats)   perplexity      bpc
Uniform baseline             4.1744        65.00     6.02
Train (your model)           1.2066         3.34     1.74
Val   (your model)           1.5012         4.49     2.17

Your model is 14.5x better than a uniform guesser at predicting the next character.
```

## Google Colab T4 Run

The runbook was also tested in Google Colab on an NVIDIA T4 GPU.

Final Colab T4 metric summary:

```text
                         nll (nats)   perplexity      bpc
Uniform baseline             4.1744        65.00     6.02
Train (your model)           1.2076         3.35     1.74
Val   (your model)           1.5058         4.51     2.17

Your model is 14.4x better than a uniform guesser at predicting the next character.
```

The MPS and T4 results are close, which suggests the notebook is reproducible
across the intended local and Colab environments.

## Generated Sample Quality

The trained model generates Shakespeare-like character-level text with visible
dialogue structure, speaker names, punctuation, and mostly English-looking word
forms. Some invented words remain, which is expected for a tiny character-level
Transformer trained for a short run.

Example characteristics:

- speaker labels such as `ANGELO:` and `DUKE VINCENTIO:`
- plausible line breaks and punctuation
- real-looking short phrases
- occasional non-words or broken long-range syntax

This quality is consistent with the final validation perplexity around `4.5`.

## Timing Notes

The training loop can optionally record wall-clock timing by storing:

- total elapsed seconds
- seconds since the previous evaluation checkpoint
- average seconds per iteration interval

For GPU or MPS timing, synchronize the device before reading the clock:

```python
if device == "cuda":
    torch.cuda.synchronize()
elif device == "mps":
    torch.mps.synchronize()
```

This makes timing more accurate because GPU and MPS operations are asynchronous.

## How to Re-run

From a clean runtime:

1. Open the notebook.
2. Select a Python environment with PyTorch installed.
3. Run all cells top to bottom.
4. Confirm the attention sanity check passes.
5. Let training finish to `max_iters`.
6. Confirm the training curve, metric table, and generated sample are visible.

The notebook chooses CUDA if available, then MPS, then CPU.
