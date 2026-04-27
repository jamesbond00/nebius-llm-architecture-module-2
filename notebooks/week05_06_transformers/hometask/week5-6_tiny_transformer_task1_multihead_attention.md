# Task 1 Guide: MultiHeadSelfAttention

This guide explains how to implement TODO 1 in
`notebooks/week05_06_transformers/hometask/tiny_transformer_lm.ipynb`.

The goal is to build a causal multi-head self-attention layer. The module takes
an input tensor `x` with shape `(B, T, C)` and returns another tensor with the
same shape. Here:

- `B` is the batch size.
- `T` is the current sequence length.
- `C` is `n_embd`, the residual stream width.

The important rule is causality: position `t` may look at positions `0..t`, but
must not look at positions after `t`.

## Mental Model

For every token position, attention asks three questions:

- Query: what am I looking for?
- Key: what information do I offer for matching?
- Value: what information should I pass forward if attended to?

Self-attention computes how strongly each token should look at previous tokens,
then mixes their value vectors using those attention weights.

Multi-head attention repeats this process several times in parallel. Each head
gets a smaller slice of the channel dimension:

```text
C = n_embd
head_size = C // n_head
```

So if `C = 192` and `n_head = 6`, each head has `head_size = 32`.

## Step 1: Store Head Metadata

The notebook already sets:

```python
self.n_head = n_head
self.head_size = n_embd // n_head
self.n_embd = n_embd
```

Keep these values in mind. Most shape mistakes in this task come from mixing up
`C`, `n_head`, and `head_size`.

Expected invariant:

```text
n_head * head_size == n_embd
```

## Step 2: Create One Q/K/V Projection

TODO 1a asks for a single linear layer that maps:

```text
(B, T, C) -> (B, T, 3C)
```

This layer produces queries, keys, and values together. Later, in `forward`, you
will split the last dimension into three equal chunks.

Use `bias=False` because the layer norm before attention can already shift the
features.

Think of the output like this:

```text
[ q channels | k channels | v channels ]
```

Each chunk has shape `(B, T, C)`.

## Step 3: Create the Output Projection

TODO 1b asks for another linear layer:

```text
(B, T, C) -> (B, T, C)
```

After attention combines all heads back into one tensor, this projection lets the
module mix information across heads before returning to the residual stream.

Bias is fine here.

## Step 4: Create the Causal Mask

TODO 1c asks for a lower-triangular matrix:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

Rows are query positions. Columns are key positions.

At row `i`, only columns `j <= i` are visible. Columns `j > i` are future
positions and must be hidden before softmax.

Register the mask as a buffer named `"mask"` so it moves with the module when
the model moves to CPU, CUDA, or MPS, but does not become a trainable parameter.

The mask is built with full shape `(block_size, block_size)`. In `forward`, use
only the current sequence part:

```python
self.mask[:T, :T]
```

This matters during generation, where `T` can be smaller than `block_size`.

## Step 5: Compute Q, K, and V in `forward`

Inside `forward`, the input shape is:

```python
B, T, C = x.shape
```

Apply the Q/K/V projection to `x`, producing `(B, T, 3C)`.

Then split that result into three tensors:

```text
q: (B, T, C)
k: (B, T, C)
v: (B, T, C)
```

Useful PyTorch tools:

- `.chunk(3, dim=-1)`
- `torch.split(..., C, dim=-1)`

Both approaches are fine.

## Step 6: Split Channels Into Heads

Each of `q`, `k`, and `v` starts as:

```text
(B, T, C)
```

You need:

```text
(B, n_head, T, head_size)
```

A good way to reason about this is two operations:

1. View the channel dimension as `n_head` groups of `head_size`.
2. Move `n_head` before `T`, because attention is computed independently per
   head.

Shape transition:

```text
(B, T, C)
-> (B, T, n_head, head_size)
-> (B, n_head, T, head_size)
```

Use `.view(...)` for the first step and `.transpose(...)` for the second.

## Step 7: Compute Attention Scores

For each head, every query position compares with every key position.

Shapes:

```text
q: (B, n_head, T, head_size)
k: (B, n_head, T, head_size)
```

To compute dot products between queries and keys, transpose the last two
dimensions of `k`:

```text
k transposed: (B, n_head, head_size, T)
```

Then matrix multiply:

```text
scores = q @ k_transposed
```

Result:

```text
scores: (B, n_head, T, T)
```

The last two dimensions mean:

```text
scores[..., query_position, key_position]
```

Scale the scores by `sqrt(head_size)`. Without this scaling, dot products grow
larger as `head_size` grows, which can make the softmax too sharp early in
training.

## Step 8: Apply the Causal Mask Before Softmax

The mask must be applied to `scores`, not to the final attention output.

Use the sliced mask:

```text
self.mask[:T, :T]
```

Where the mask is zero, replace the corresponding score with negative infinity.
After softmax, those entries become probability zero.

Expected effect for a four-token sequence:

```text
token 0 can attend to: 0
token 1 can attend to: 0, 1
token 2 can attend to: 0, 1, 2
token 3 can attend to: 0, 1, 2, 3
```

If the validation cell says changing the last input changed earlier outputs,
the mask is probably wrong, reversed, sliced incorrectly, or applied after
softmax.

## Step 9: Convert Scores to Attention Weights

Apply softmax over the last dimension:

```text
scores: (B, n_head, T, T)
att:    (B, n_head, T, T)
```

The last dimension corresponds to key positions. For each query position, the
weights over all allowed key positions should sum to 1.

Then apply attention dropout:

```python
self.attn_dropout(...)
```

Dropout is active during training and disabled in `.eval()` mode.

## Step 10: Mix the Values

Now multiply attention weights by values:

```text
att: (B, n_head, T, T)
v:   (B, n_head, T, head_size)
```

Matrix multiply:

```text
y: (B, n_head, T, head_size)
```

For each query position, this gives a weighted average of the allowed value
vectors.

## Step 11: Merge Heads Back Together

The attention output is currently split by head:

```text
(B, n_head, T, head_size)
```

The rest of the transformer expects the residual stream shape:

```text
(B, T, C)
```

Reverse the earlier transformation:

```text
(B, n_head, T, head_size)
-> (B, T, n_head, head_size)
-> (B, T, C)
```

After `transpose`, use `.contiguous()` before `.view(...)`. Transposed tensors
often have non-contiguous memory layout, and `.view(...)` expects a compatible
layout.

## Step 12: Apply Output Projection and Residual Dropout

Finally:

1. Pass `y` through the output projection.
2. Pass the result through residual dropout.
3. Return the result.

The returned tensor must have shape:

```text
(B, T, C)
```

Do not add the residual connection inside this module. The `Block` module will
handle residual connections later.

## Checklist

Before running the attention check cell, verify:

- `self.qkv` maps `n_embd` to `3 * n_embd`.
- `self.out_proj` maps `n_embd` to `n_embd`.
- The mask is lower-triangular and registered as a buffer.
- The mask is sliced with `[:T, :T]` inside `forward`.
- Q, K, and V are reshaped to `(B, n_head, T, head_size)`.
- Attention scores have shape `(B, n_head, T, T)`.
- The mask is applied before softmax.
- Softmax uses `dim=-1`.
- Heads are merged back to `(B, T, C)`.
- The final return uses output projection and residual dropout.
- The `NotImplementedError` line is removed after implementation.

## Common Bugs

**Wrong mask direction**

If future tokens are visible and past tokens are hidden, the triangular mask is
upside down. Use lower-triangular masking for causal language modelling.

**Mask applied after softmax**

This breaks probability normalization and may not fully prevent future
information flow. Mask scores before softmax.

**Softmax over the wrong dimension**

Softmax should run over key positions, which are in the last dimension of
`scores`.

**Forgot to transpose heads**

If attention scores do not become `(B, n_head, T, T)`, check the reshape and
transpose steps.

**Forgot `.contiguous()` before `.view()`**

After transposing back from `(B, n_head, T, head_size)` to
`(B, T, n_head, head_size)`, call `.contiguous()` before viewing as `(B, T, C)`.

**Added the residual connection in the attention module**

This module should only return the transformed attention output. The surrounding
transformer block adds it to the original input.

## How to Debug Shapes

Temporarily print shapes after each major step:

```python
print("qkv", qkv.shape)
print("q", q.shape)
print("scores", scores.shape)
print("att", att.shape)
print("y before merge", y.shape)
print("y after merge", y.shape)
```

Remove these prints once the check passes.

The check cell tests three important properties:

- Output shape equals input shape.
- Changing the final token does not affect earlier outputs.
- Shorter sequences still work because the mask is sliced to the current `T`.
