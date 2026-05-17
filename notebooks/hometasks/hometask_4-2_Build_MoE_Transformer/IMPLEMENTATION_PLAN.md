# Build MoE Transformer: Implementation Plan

This guide explains how to complete
`4_2_tiny_moe_lm_hometask_Alex_20260517.ipynb`.

The notebook starts from the tiny Transformer language model and asks you to add
two important upgrades:

1. **RoPE**: rotary position embeddings inside attention.
2. **MoE**: a sparse mixture-of-experts replacement for the normal feed-forward
   layer.

The final model is still a small character-level language model trained on Tiny
Shakespeare. It predicts the next character from the previous `block_size`
characters.

## Mental Model

### What A Transformer Block Does

Each block has two main sublayers:

```text
input tokens
  -> self-attention: tokens exchange information with previous tokens
  -> feed-forward / MoE: each token is processed independently
  -> output tokens
```

Attention answers: "Which earlier tokens should this token look at?"

The feed-forward layer answers: "After gathering context, how should this token's
features be transformed?"

In this homework, attention gets RoPE and the feed-forward layer becomes MoE.

### Why RoPE Exists

Attention itself does not know token order. Without position information,
`KING` at position 1 and `KING` at position 100 look the same.

The previous tiny Transformer probably used learned position embeddings:

```text
token embedding + position embedding
```

RoPE is different. It does not add a position vector to the token. Instead, it
rotates the query and key vectors by a position-dependent angle before computing
attention scores.

The intuition:

- every token still has a content vector
- the query and key are rotated depending on where the token appears
- dot products between rotated queries and keys naturally include relative
  position information

Only `q` and `k` are rotated. Values `v` are not rotated.

### Why MoE Exists

A normal feed-forward layer sends every token through the same network:

```text
token -> same FFN -> output
```

An MoE layer has several feed-forward networks called experts:

```text
token -> router -> top-k experts -> weighted expert outputs -> output
```

The router decides which experts should process each token. With `top_k = 2`,
each token is sent to two experts, not all experts.

This gives the model more total parameters without using all of them for every
token. That is the core sparsity idea.

## Notebook TODO Map

Complete the TODOs in this order:

1. `RoPE`
2. `MultiHeadSelfAttentionRope`
3. `MoE`
4. `TinyMoeLM`
5. Run the provided unit tests and sanity checks
6. Train and inspect loss/drop-rate curves

This order is important because later components depend on earlier ones.

## Step 1: Implement `RoPE`

### Expected Input Shapes

The attention module should produce query and key tensors shaped like:

```text
q, k: (batch, num_heads, seq_len, head_dim)
```

RoPE should return tensors with exactly the same shape.

### What To Precompute

In `__init__`, create cached cosine and sine tensors for all positions up to
`max_seq_len`.

The notebook starter code shows:

```python
emb = torch.arange(0, dim // 2)
self.register_buffer("cos_cached", emb.cos())
self.register_buffer("sin_cached", emb.sin())
```

Treat that as a placeholder, not a correct implementation. It only creates one
row of angles for half the head dimension. Real RoPE needs angles for every
sequence position and for the full head dimension after frequency duplication.

Use the standard inverse-frequency formula:

```python
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
```

For each position `m`, compute:

```python
freqs = position * inv_freq
```

The notebook's `rotate_half` test expects the "split-half" style:

```text
[x1, x2] -> [-x2, x1]
```

Because of that, duplicate the frequencies so the final cached tensors have
last dimension `dim`:

```python
emb = torch.cat((freqs, freqs), dim=-1)
```

Then register:

```python
self.register_buffer("cos_cached", emb.cos())
self.register_buffer("sin_cached", emb.sin())
```

Good cached shape:

```text
cos_cached, sin_cached: (max_seq_len, head_dim)
```

### Implement `rotate_half`

For a tensor whose last dimension is split into two equal halves:

```python
def rotate_half(self, x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
```

Example:

```text
[1, 2, 3, 4] -> [-3, -4, 1, 2]
```

This matches the unit test.

### Implement `forward`

In `forward(q, k)`:

1. Read `seq_len = q.size(-2)`.
2. Slice the caches to that sequence length.
3. Add broadcast dimensions for batch and heads.
4. Apply the rotation formula to both `q` and `k`.

The cache shape starts as:

```text
(seq_len, head_dim)
```

For broadcasting with:

```text
(batch, num_heads, seq_len, head_dim)
```

reshape with:

```python
cos = self.cos_cached[:seq_len].to(q.device).unsqueeze(0).unsqueeze(0)
sin = self.sin_cached[:seq_len].to(q.device).unsqueeze(0).unsqueeze(0)
```

Then:

```python
q_rot = (q * cos) + (self.rotate_half(q) * sin)
k_rot = (k * cos) + (self.rotate_half(k) * sin)
return q_rot, k_rot
```

### RoPE Self-Check

The norm test should pass because rotation preserves vector length. If the norm
changes, common causes are:

- cached `cos`/`sin` have the wrong last dimension
- broadcasting is wrong
- `rotate_half` does not match `[-x2, x1]`

### Complete `RoPE` Class Implementation

Paste this into the notebook cell that currently contains the incomplete
`RoPE` class:

```python
class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: int = 10000.0):
        super().__init__()

        assert dim % 2 == 0, "RoPE head dimension must be even."

        # One frequency per pair of hidden dimensions.
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

        # Positions: 0, 1, 2, ..., max_seq_len - 1
        positions = torch.arange(max_seq_len, dtype=torch.float32)

        # freqs[m, i] = position m * inverse frequency i
        freqs = torch.einsum("m,i->mi", positions, inv_freq)

        # The notebook's rotate_half implementation splits the vector into two
        # halves, so duplicate frequencies to match the full head dimension.
        emb = torch.cat((freqs, freqs), dim=-1)

        # Buffers move with the module across CPU/GPU, but are not trainable.
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the last dimension from [x1, x2] to [-x2, x1]."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)

        # q and k have shape (batch, heads, seq_len, head_dim).
        # Cached cos/sin have shape (max_seq_len, head_dim), so add two leading
        # singleton dimensions for broadcasting over batch and heads.
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)

        return q_rot, k_rot
```

You do not need to manually call `.to(q.device)` here because registered buffers
move with the module when you call `.to(device)` on the full model. Adding
`.to(q.device)` is also acceptable, but it is usually cleaner to rely on
PyTorch's buffer behavior.

## Step 2: Implement `MultiHeadSelfAttentionRope`

This is the same causal multi-head self-attention as before, except query and
key pass through RoPE before the attention scores are computed.

### Components To Create In `__init__`

Use:

```python
self.n_head = n_head
self.head_size = n_embd // n_head
self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
self.proj = nn.Linear(n_embd, n_embd)
self.attn_dropout = nn.Dropout(dropout)
self.resid_dropout = nn.Dropout(dropout)
self.rope = RoPE(self.head_size, max_seq_len=block_size)
```

Also create a causal mask:

```python
self.register_buffer(
    "tril",
    torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
)
```

### Forward Pass Shape Walkthrough

Input:

```text
x: (B, T, C)
```

Where:

- `B` = batch size
- `T` = sequence length
- `C` = embedding size

Project once and split:

```python
q, k, v = self.qkv(x).split(C, dim=-1)
```

Reshape each to multi-head format:

```python
q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
```

Now:

```text
q, k, v: (B, n_head, T, head_size)
```

Apply RoPE:

```python
q, k = self.rope(q, k)
```

Compute causal attention:

```python
wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
wei = wei.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
wei = self.attn_dropout(wei)
out = wei @ v
```

Merge heads:

```python
out = out.transpose(1, 2).contiguous().view(B, T, C)
out = self.resid_dropout(self.proj(out))
return out
```

### Common Attention Bugs

- Forgetting `.contiguous()` before `.view(...)` after a transpose.
- Applying RoPE before reshaping into heads.
- Rotating `v`. RoPE should only rotate `q` and `k`.
- Forgetting the causal mask, which lets the model see future characters.

### Complete `MultiHeadSelfAttentionRope` Class Implementation

Paste this into the notebook cell that currently contains the incomplete
`MultiHeadSelfAttentionRope` class:

```python
class MultiHeadSelfAttentionRope(nn.Module):
    """Causal multi-head self-attention with RoPE."""
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()

        assert n_embd % n_head == 0, "n_embd must be divisible by n_head."

        self.n_head = n_head
        self.head_size = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.rope = RoPE(self.head_size, max_seq_len=block_size)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=-1)

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        q, k = self.rope(q, k)

        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.proj(out))

        return out
```

If your notebook currently has:

```python
q_r, k_r = self.rope(q, k)
wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
```

change it to:

```python
q, k = self.rope(q, k)
wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
```

Otherwise RoPE is computed and then ignored.

## Step 3: Implement `MoE`

### What The MoE Layer Replaces

The normal block had:

```python
x = x + self.ffwd(self.ln2(x))
```

The MoE block uses:

```python
moe_out, drop_rate = self.moe(self.ln2(x))
x = x + moe_out
```

The MoE output must have the same shape as its input:

```text
(batch, seq_len, n_embd)
```

### Components To Create In `__init__`

Store the configuration:

```python
self.num_experts = num_experts
self.top_k = top_k
self.capacity_factor = capacity_factor
```

Create the router:

```python
self.router = nn.Linear(n_embd, num_experts, bias=False)
```

Create the experts:

```python
self.experts = nn.ModuleList([
    FeedForward(n_embd, exp_hid_dim, dropout)
    for _ in range(num_experts)
])
```

The notebook defines `exp_hid_dim = hid_dim // 2` globally. That is intentional:
there are more experts, but each expert is smaller.

### Forward Pass Overview

Input:

```text
x: (B, T, C)
```

Flatten tokens:

```python
B, T, C = x.shape
x_flat = x.view(B * T, C)
total_tokens = x_flat.size(0)
```

Each row in `x_flat` is one token representation.

### Capacity Calculation

Each token is assigned to `top_k` experts, so total expert assignments are:

```text
total_tokens * top_k
```

Average assignments per expert:

```text
(total_tokens * top_k) / num_experts
```

Capacity allows a bit more than average:

```python
capacity = int((total_tokens * self.top_k / self.num_experts) * self.capacity_factor)
capacity = max(capacity, 1)
```

The `max` avoids zero capacity for tiny test batches.

### Routing Logic

Follow the notebook's required order exactly:

```python
router_logits = self.router(x_flat)
router_probs = F.softmax(router_logits, dim=-1)
topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
```

Shapes:

```text
router_logits: (total_tokens, num_experts)
router_probs:  (total_tokens, num_experts)
topk_weights:  (total_tokens, top_k)
topk_indices:  (total_tokens, top_k)
```

Normalize the top-k weights because the selected probabilities will usually sum
to less than `1.0` after dropping the unselected experts.

### Dispatch To Experts

Create an output accumulator:

```python
out_flat = torch.zeros_like(x_flat)
total_dropped = 0
```

For each expert id:

```python
for expert_id, expert in enumerate(self.experts):
    token_idx, choice_idx = torch.where(topk_indices == expert_id)
```

Here:

- `token_idx` tells which flattened tokens selected this expert
- `choice_idx` tells whether this was their first or second selected expert

If too many tokens selected this expert:

```python
num_assigned = token_idx.numel()
if num_assigned > capacity:
    total_dropped += num_assigned - capacity
    token_idx = token_idx[:capacity]
    choice_idx = choice_idx[:capacity]
```

Process only the kept tokens:

```python
if token_idx.numel() == 0:
    continue

expert_input = x_flat[token_idx]
expert_output = expert(expert_input)
weights = topk_weights[token_idx, choice_idx].unsqueeze(-1)
out_flat.index_add_(0, token_idx, expert_output * weights)
```

`index_add_` matters because the same token receives contributions from two
experts. You are accumulating both contributions into the same output row.

Finally:

```python
out = out_flat.view(B, T, C)
drop_rate = total_dropped / (total_tokens * self.top_k)
return out, drop_rate
```

Returning `drop_rate` as a Python float is fine for the tests. Returning a scalar
tensor is also fine, but keep it on the right device if you do that.

### Why Token Dropping Happens

Imagine a batch has 8192 token positions and `top_k = 2`.

That creates:

```text
8192 * 2 = 16384 expert assignments
```

With 8 experts, perfect balance would be:

```text
16384 / 8 = 2048 assignments per expert
```

With `capacity_factor = 1.25`, each expert can process:

```text
2048 * 1.25 = 2560 assignments
```

If one expert receives 3000 assignments, it processes 2560 and drops 440.

The drop rate is:

```text
dropped_assignments / all_assignments
```

High drop rate means the router is sending too many tokens to the same experts.

### Common MoE Bugs

- Using logits directly instead of softmax probabilities.
- Forgetting to normalize only the selected top-k weights.
- Replacing output rows instead of accumulating with `index_add_`.
- Returning output in flattened shape instead of `(B, T, C)`.
- Counting dropped tokens instead of dropped assignments.
- Creating `out_flat` on CPU while `x` is on GPU.

### Complete `MoE` Class Implementation

Paste this into the notebook cell that currently contains the incomplete `MoE`
class:

```python
class MoE(nn.Module):
    def __init__(
        self,
        n_embd: int,
        dropout: float,
        num_experts: int,
        top_k: int,
        capacity_factor: float,
    ):
        super().__init__()

        assert top_k <= num_experts, "top_k cannot be larger than num_experts."

        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.router = nn.Linear(n_embd, num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(n_embd, exp_hid_dim, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        B, T, C = x.shape

        # Treat every position in every sequence as one routable token.
        x_flat = x.view(B * T, C)
        total_tokens = x_flat.size(0)

        # Capacity is measured in expert assignments, not unique tokens.
        capacity = int(
            (total_tokens * self.top_k / self.num_experts)
            * self.capacity_factor
        )
        capacity = max(capacity, 1)

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        topk_weights, topk_indices = torch.topk(
            router_probs,
            self.top_k,
            dim=-1,
        )

        # After selecting top-k, renormalize so each token's selected expert
        # weights sum to 1.
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        out_flat = torch.zeros_like(x_flat)
        total_dropped = 0

        for expert_id, expert in enumerate(self.experts):
            # token_idx: which flattened token selected this expert
            # choice_idx: whether this expert was top-1, top-2, etc. for it
            token_idx, choice_idx = torch.where(topk_indices == expert_id)

            num_assigned = token_idx.numel()
            if num_assigned == 0:
                continue

            if num_assigned > capacity:
                total_dropped += num_assigned - capacity
                token_idx = token_idx[:capacity]
                choice_idx = choice_idx[:capacity]

            expert_input = x_flat[token_idx]
            expert_output = expert(expert_input)
            expert_weight = topk_weights[token_idx, choice_idx].unsqueeze(-1)

            # A token can be sent to multiple experts, so add contributions.
            out_flat.index_add_(0, token_idx, expert_output * expert_weight)

        out = out_flat.view(B, T, C)
        drop_rate = total_dropped / (total_tokens * self.top_k)

        return out, drop_rate
```

The key line is:

```python
out_flat.index_add_(0, token_idx, expert_output * expert_weight)
```

Use addition instead of assignment because each token has `top_k` expert outputs.
For `top_k = 2`, the final token output is:

```text
weight_1 * expert_1(token) + weight_2 * expert_2(token)
```

If you assign instead of add, the second expert can overwrite the first expert's
contribution.

## Step 4: Implement `TinyMoeLM`

This model is almost the same as the previous tiny language model.

The key difference:

- keep token embeddings
- remove learned position embeddings
- each block returns both `x` and `drop_rate`
- average drop rates across layers

### Components To Create In `__init__`

Use:

```python
self.block_size = block_size
self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
self.blocks = nn.ModuleList([
    BlockMoe(n_embd, n_head, block_size, dropout)
    for _ in range(n_layer)
])
self.ln_f = nn.LayerNorm(n_embd)
self.lm_head = nn.Linear(n_embd, vocab_size)
```

Do not create `position_embedding_table`. RoPE replaces it.

### Forward Pass

Input:

```text
idx: (B, T)
targets: optional (B, T)
```

Check the sequence length:

```python
B, T = idx.shape
assert T <= self.block_size
```

Embed tokens:

```python
x = self.token_embedding_table(idx)
```

Run blocks and collect drop rates:

```python
layer_drop_rates = []
for block in self.blocks:
    x, drop_rate = block(x)
    layer_drop_rates.append(drop_rate)
```

Finish:

```python
x = self.ln_f(x)
logits = self.lm_head(x)
avg_drop_rate = sum(layer_drop_rates) / len(layer_drop_rates)
```

Compute loss only if targets are provided:

```python
if targets is None:
    loss = None
else:
    B, T, C_vocab = logits.shape
    loss = F.cross_entropy(logits.view(B * T, C_vocab), targets.view(B * T))
```

Return:

```python
return logits, loss, avg_drop_rate
```

### Implement `generate`

The sanity check calls:

```python
_test_model.generate(_ctx, max_new_tokens=5)
```

So `TinyMoeLM` must include a `generate` method.

Use the standard autoregressive loop:

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.block_size:]
        logits, _, _ = self(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

The crop is required because the sanity check passes a context longer than
`block_size`.

## Step 5: Verification Plan

Run checks in this order.

### 1. RoPE Unit Tests

Expected result:

- `rotate_half` maps `[1, 2, 3, 4]` to `[-3, -4, 1, 2]`
- rotated query/key shapes match input shapes
- query vector norms are preserved

### 2. MoE Unit Tests

Expected result:

- output shape equals input shape
- no `NaN`s
- large capacity factor gives `drop_rate == 0.0`
- tiny capacity factor gives high drop rate

### 3. Full Model Sanity Check

Expected result:

- logits shape is `(batch_size, block_size, vocab_size)`
- initial loss is close to `log(vocab_size)`
- generation returns valid token ids
- long context generation works because `generate` crops context

### 4. Training

Expected behavior:

- train loss should fall below about `1.2`
- validation loss should be slightly below about `1.5`
- drop rate should stay under about `15%` by the end
- drop rate above `20%` after the first 100 iterations is suspicious

## Debugging Checklist

### If RoPE Tests Fail

Check:

- `head_dim` is even
- `cos_cached` and `sin_cached` have shape `(max_seq_len, head_dim)`
- `rotate_half` uses `x.chunk(2, dim=-1)`
- `cos` and `sin` are sliced to current `seq_len`
- cache tensors are moved to the same device as `q` and `k`

### If Attention Output Shape Is Wrong

Check:

- `n_embd % n_head == 0`
- after projection, `q`, `k`, `v` are reshaped to `(B, n_head, T, head_size)`
- final output is reshaped back to `(B, T, C)`

### If MoE Has NaNs

Check:

- softmax is applied before top-k
- top-k weights are normalized with `keepdim=True`
- no division by zero
- output tensor starts as `torch.zeros_like(x_flat)`

### If Drop Rate Is Always Zero

Check:

- capacity is calculated from `total_tokens * top_k`, not just `total_tokens`
- token dropping compares `num_assigned > capacity`
- `total_dropped` is incremented before slicing

### If Drop Rate Is Too High During Training

Possible causes:

- router is collapsing to one or two experts
- capacity formula is too small
- `capacity_factor` was changed accidentally
- top-k indices are being interpreted incorrectly

For this homework, keep the provided hyperparameters unless you are debugging.

## Minimal Code Skeleton

This is not a full copy-paste solution, but it shows the intended structure.

```python
class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k):
        seq_len = q.size(-2)
        cos = self.cos_cached[:seq_len].to(q.device).unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].to(q.device).unsqueeze(0).unsqueeze(0)
        return q * cos + self.rotate_half(q) * sin, k * cos + self.rotate_half(k) * sin
```

```python
class MoE(nn.Module):
    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(B * T, C)
        total_tokens = x_flat.size(0)
        capacity = int((total_tokens * self.top_k / self.num_experts) * self.capacity_factor)
        capacity = max(capacity, 1)

        probs = F.softmax(self.router(x_flat), dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        out_flat = torch.zeros_like(x_flat)
        total_dropped = 0

        for expert_id, expert in enumerate(self.experts):
            token_idx, choice_idx = torch.where(indices == expert_id)
            assigned = token_idx.numel()
            if assigned > capacity:
                total_dropped += assigned - capacity
                token_idx = token_idx[:capacity]
                choice_idx = choice_idx[:capacity]
            if token_idx.numel() == 0:
                continue
            expert_out = expert(x_flat[token_idx])
            expert_weight = weights[token_idx, choice_idx].unsqueeze(-1)
            out_flat.index_add_(0, token_idx, expert_out * expert_weight)

        drop_rate = total_dropped / (total_tokens * self.top_k)
        return out_flat.view(B, T, C), drop_rate
```

Use this skeleton to guide your implementation, but still write the complete
classes in the notebook so all tests and training cells can run.
