# LoRA from Scratch on GPT-2

This folder contains the LoRA homework notebook:

```text
homework_lora.ipynb
```

The notebook is both an implementation exercise and a small experiment. You
write a minimal LoRA layer by hand, plug it into GPT-2, fine-tune only the LoRA
weights on TinyShakespeare, then compare your implementation with Hugging Face
PEFT.

## What This Homework Is Doing

GPT-2 is already a trained language model. Full fine-tuning would update all of
its parameters, which is expensive and creates a full-size model checkpoint.
LoRA takes a different route:

- keep the original GPT-2 weights frozen
- add a small trainable low-rank update to selected linear layers
- train only those new low-rank weights
- save only the small adapter after training

In this notebook, the base model stays almost entirely fixed. The trainable part
is the LoRA adapter. That is why the trainable parameter count should be around
0.6-0.7% of GPT-2 small, instead of 100%.

The target behavior is style adaptation. GPT-2 already knows general English.
The LoRA adapter nudges it toward Shakespeare-like text by training on
TinyShakespeare.

## Key Ideas

### Frozen Base Model

A frozen parameter has `requires_grad=False`. PyTorch will still use it in the
forward pass, but it will not compute gradients for it or update it during the
optimizer step.

For LoRA, this matters because GPT-2's original knowledge stays in the base
weights. The adapter learns a small task-specific correction.

### Low-Rank Update

A normal linear layer applies:

```text
y = x @ W.T + b
```

LoRA adds a trainable residual update:

```text
y = base(x) + scaling * dropout(x) @ A.T @ B.T
```

The update is low-rank because it is represented as two smaller matrices:

- `A`: shape `(r, in_features)`
- `B`: shape `(out_features, r)`

Instead of learning a full `(out_features, in_features)` matrix, LoRA learns
only:

```text
r * (in_features + out_features)
```

parameters.

### Rank `r`

The rank controls the size and capacity of the adapter.

- smaller `r`: fewer trainable parameters, cheaper training, less capacity
- larger `r`: more trainable parameters, more capacity, larger adapter

The homework uses `r=8`, which is small enough to be cheap but large enough to
show a visible style shift.

### Alpha and Scaling

The notebook uses:

```python
scaling = alpha / r
```

`alpha` controls the strength of the LoRA update. With `r=8` and `alpha=16`, the
scaling factor is `2.0`.

### Why Initialize `B` to Zeros?

At initialization, the LoRA update should be exactly zero:

```text
B @ A = 0
```

because `B` starts as all zeros. That means the wrapped model produces the same
outputs as the original GPT-2 before fine-tuning. This is a useful safety
property: adding LoRA should not change model behavior until training starts.

### Dropout in the LoRA Branch

Dropout is applied only before the LoRA branch:

```text
dropout(x) @ A.T @ B.T
```

This regularizes the adapter. The frozen base path still runs normally.

### Adapter Checkpoint

The adapter checkpoint contains only parameters whose names include `lora_`.
That file is much smaller than a full GPT-2 checkpoint.

To reuse the adapter, you:

1. load a fresh GPT-2
2. inject LoRA modules with the same configuration
3. load the saved `lora_A` and `lora_B` tensors

The notebook verifies this by checking that logits from the trained model and a
fresh model plus loaded adapter match.

### Perplexity

Perplexity is a language-model metric where lower is better. It is the
exponentiated average cross-entropy loss.

In this homework, you measure:

- Shakespeare validation perplexity
- Pride & Prejudice control perplexity

The Shakespeare number should usually improve after fine-tuning. The control
number helps show whether the model became more specialized and less general.

### PEFT

PEFT is Hugging Face's production-grade library for parameter-efficient
fine-tuning. The homework first asks you to implement LoRA by hand so the
mechanics are visible. Then it repeats the experiment with PEFT so you can
compare your implementation with a standard library implementation.

## Notebook Walkthrough

### 1. Setup

The notebook imports PyTorch, Transformers, PEFT, and basic utilities. It picks
the best available device in this order:

```text
CUDA -> MPS -> CPU
```

On an Apple Silicon machine, seeing `device: mps` is expected.

### 2. Inspect and Prepare GPT-2

Hugging Face GPT-2 uses `Conv1D` modules for many projections. These are
functionally linear layers, but their weight layout is different from
`nn.Linear`.

The homework provides a helper that converts GPT-2's `Conv1D` modules to
ordinary `nn.Linear` modules. This keeps the custom LoRA implementation focused
on one familiar layer type.

### 3. Baseline Generation

Before training, the notebook generates text from unmodified GPT-2 using prompts
such as:

```text
ROMEO:
To be, or not to be,
Once upon a time in fair Verona,
```

Save these outputs in the notebook. They are the baseline for qualitative
comparison after fine-tuning.

### 4. Implement `LoRALinear`

`LoRALinear` wraps an existing `nn.Linear`. The wrapped layer has two paths:

```text
output = frozen base path + trainable LoRA path
```

The base path is the original GPT-2 projection. The LoRA path is the small
adapter you train.

The target forward pass is:

```python
base(x) + scaling * dropout(x) @ lora_A.T @ lora_B.T
```

where:

```python
scaling = alpha / r
```

#### What to Add in `__init__`

First, freeze the original linear layer:

```python
self.base.weight.requires_grad = False
if self.base.bias is not None:
    self.base.bias.requires_grad = False
```

This is the whole point of LoRA. The original model remains fixed, and the
optimizer will only update the new adapter parameters.

Next, create the two LoRA matrices:

```python
factory_kwargs = {
    "device": base.weight.device,
    "dtype": base.weight.dtype,
}

self.lora_A = nn.Parameter(torch.empty((r, in_features), **factory_kwargs))
self.lora_B = nn.Parameter(torch.empty((out_features, r), **factory_kwargs))
```

Use the base layer's device and dtype. If the base model is on MPS or CUDA but
the LoRA tensors are accidentally created on CPU, the first forward pass will
fail with a device mismatch.

Initialize them like this:

```python
nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
nn.init.zeros_(self.lora_B)
```

`A` starts with ordinary random values so it can learn useful directions.
`B` starts at zero so the whole LoRA branch starts at zero:

```text
B @ A = 0
```

That makes the wrapper behavior-preserving at initialization:

```text
LoRALinear(base)(x) == base(x)
```

This is why the first sanity check expects the wrapped layer and the base layer
to produce identical outputs before training.

Finally, create the dropout module:

```python
self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
```

`nn.Identity()` keeps the code simple because the forward pass can always call
`self.lora_dropout(x)` whether dropout is enabled or not.

#### Tensor Shapes in `forward`

For GPT-style hidden states, `x` usually has shape:

```text
batch_size x sequence_length x in_features
```

The LoRA branch shape flow is:

```text
dropout(x):                    batch x seq x in_features
lora_A.T:                      in_features x r
dropout(x) @ lora_A.T:         batch x seq x r
lora_B.T:                      r x out_features
... @ lora_B.T:                batch x seq x out_features
```

That final shape matches `base(x)`, so the two paths can be added.

The implementation should be:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    lora_update = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
    return self.base(x) + self.scaling * lora_update
```

This expression uses normal PyTorch batched matrix multiplication. PyTorch
handles the leading `batch x seq` dimensions automatically.

#### Why `A` Then `B`?

The LoRA update is conceptually:

```text
delta_W = B @ A
```

`delta_W` has the same shape as the base linear weight:

```text
out_features x in_features
```

But the forward pass applies the matrices to activations, so the order looks
transposed:

```text
x @ A.T @ B.T
```

This is the same operation as applying `delta_W` through a linear layer:

```text
x @ (B @ A).T
```

Remember it this way:

- merged weight view: `B @ A`
- activation view: `x @ A.T @ B.T`

#### Implement `merged_weight`

`merged_weight()` should return the effective inference-time weight:

```python
return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)
```

This is useful because LoRA does not have to stay as two separate matrices at
inference time. The low-rank update can be merged into the frozen base weight.

The notebook checks this by comparing:

```python
lora(x)
```

against:

```python
F.linear(x, lora.merged_weight(), base.bias)
```

Those should match because both compute the same effective linear operation.

#### Complete Implementation Shape

The filled class should follow this structure:

```python
class LoRALinear(nn.Module):
    """Frozen linear layer + trainable low-rank residual."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear), "LoRALinear only wraps nn.Linear in this homework."
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = base.in_features
        out_features = base.out_features

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        factory_kwargs = {
            "device": base.weight.device,
            "dtype": base.weight.dtype,
        }
        self.lora_A = nn.Parameter(torch.empty((r, in_features), **factory_kwargs))
        self.lora_B = nn.Parameter(torch.empty((out_features, r), **factory_kwargs))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_update = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return self.base(x) + self.scaling * lora_update

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)
```

#### What the Sanity Checks Prove

The provided sanity checks are not just mechanical. Each one protects against a
common LoRA bug:

- `lora_A` and `lora_B` on the same device as `base.weight`: prevents CPU/GPU/MPS
  mismatch errors.
- LoRA output equals base output at initialization: confirms `B` was initialized
  to zeros and the residual starts inactive.
- Perturbing `lora_B` changes the output: confirms the LoRA branch actually
  participates in the forward pass.
- `merged_weight()` matches `forward()`: confirms the matrix orientation is
  correct.

If these pass, the layer is ready to inject into GPT-2.

### 5. Inject LoRA into GPT-2

The notebook targets modules named:

```python
("c_attn", "c_proj")
```

For GPT-2 small, this should wrap 36 layers:

```text
12 transformer blocks * (1 attn.c_attn + 1 attn.c_proj + 1 mlp.c_proj)
```

Only `lora_A` and `lora_B` should remain trainable.

Expected trainable parameter math with `r=8`:

```text
attn.c_attn: 768 -> 2304  gives 8 * (768 + 2304)  = 24,576
attn.c_proj: 768 -> 768   gives 8 * (768 + 768)   = 12,288
mlp.c_proj:  3072 -> 768  gives 8 * (3072 + 768)  = 30,720
per block: 67,584
12 blocks: 811,008
```

That is about 0.65% of GPT-2 small's roughly 124M parameters.

### 6. Train on TinyShakespeare

The notebook downloads TinyShakespeare, tokenizes the text, splits it into
fixed-size chunks, and trains the LoRA adapter as a causal language model.

The optimizer should receive only trainable parameters:

```python
[p for p in model.parameters() if p.requires_grad]
```

This is important because optimizer state can be a major memory cost on larger
models.

### 7. Compare Before and After

After training, generate from the same prompts and compare against the baseline.
Look for concrete changes:

- Shakespeare-like diction
- character names
- dialogue formatting
- line breaks
- punctuation style
- archaic words

Then record the before/after perplexity values for both the Shakespeare
validation set and the Pride & Prejudice control text.

### 8. Save and Load the Adapter

Save only the LoRA parameters:

```text
keys containing "lora_"
```

Then load those parameters into a fresh GPT-2 with the same LoRA injection
configuration. The notebook sets both models to eval mode before comparing
logits, because dropout must be disabled for deterministic comparison.

### 9. Compare with PEFT

The PEFT run should use the same LoRA configuration:

```python
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],
    bias="none",
)
```

Train it with the same `TrainCfg`, then compare:

- trainable parameter count
- Shakespeare validation perplexity
- Pride & Prejudice control perplexity
- generated samples

Small differences are normal. PEFT handles GPT-2's original `Conv1D` modules
directly, while the hand-written implementation uses converted `nn.Linear`
modules.

## Recommended Runtime Strategy

### Phase 1: Local MPS Smoke Test

Use the M3 Pro/MPS machine first to catch implementation bugs cheaply.

Run these parts locally:

- setup and imports
- GPT-2 load
- `Conv1D` to `nn.Linear` conversion
- `LoRALinear` sanity checks
- LoRA injection and trainable-parameter count
- one short training smoke run if desired
- adapter save/load round-trip logic

For a quick local smoke run, use:

```python
cfg = TrainCfg(epochs=1, batch_size=4, lr=3e-4)
```

This may not produce final-quality metrics, but it should prove the code works.

MPS is useful for correctness checks, but it is usually slower and less
predictable than CUDA for this kind of Transformer training.

### Phase 2: Cloud GPU Final Run

Use RunPod, Hyperbolic, or Colab T4 for the final executed notebook.

A T4-class CUDA GPU is enough for GPT-2 small LoRA fine-tuning. Larger GPUs will
finish faster but are not required.

Use the intended final config unless memory forces a smaller batch size:

```python
cfg = TrainCfg(epochs=2, batch_size=8, lr=3e-4)
```

The final saved notebook should contain:

- baseline generations
- hand-written LoRA training logs
- post-training generations
- Shakespeare/control perplexity before and after
- adapter file size and round-trip assertion
- PEFT trainable parameter count
- PEFT training logs
- PEFT PPL comparison
- written answers for Q1-Q5

### Phase 3: Colab Fallback

If setting up a RunPod or Hyperbolic VM takes too long, use Colab with a T4 GPU.
The notebook already includes a Colab install cell.

Colab is less configurable than a rented VM, but it is sufficient for this
homework.

## Cloud VM Setup

These steps are intentionally generic so they work on RunPod, Hyperbolic, or a
similar GPU VM.

1. Start a PyTorch/CUDA image.

2. Clone the repository:

```bash
git clone https://github.com/jamesbond00/nebius-llm-architecture-module-2.git
cd nebuis-llm-architecture-module-2
```

3. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

4. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

5. Confirm CUDA is visible:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

6. Start Jupyter:

```bash
jupyter lab --ip 0.0.0.0 --port 8888 --no-browser
```

7. Open the notebook:

```text
notebooks/hometasks/hometask_4-1_LoRA/homework_lora.ipynb
```

8. Run all cells top to bottom and save the completed notebook.

## Validation Checklist

Before training:

- [ ] `LoRALinear` matches the base layer at initialization.
- [ ] Perturbing `lora_B` changes the output.
- [ ] `merged_weight()` matches the forward pass.
- [ ] LoRA injection wraps exactly 36 modules.
- [ ] Trainable params are around 811,008.
- [ ] Trainable fraction is about 0.65%.
- [ ] All model parameters are on one device.

After hand-written LoRA training:

- [ ] Baseline generations are visible.
- [ ] Post-training generations are visible.
- [ ] Shakespeare validation PPL before/after is recorded.
- [ ] Pride & Prejudice control PPL before/after is recorded.
- [ ] Q1-Q3 are answered using actual outputs and metrics.

Adapter check:

- [ ] Adapter file is saved.
- [ ] Adapter file size is printed.
- [ ] Fresh GPT-2 plus loaded adapter reproduces trained-model logits.

PEFT comparison:

- [ ] PEFT config uses `r=8`, `lora_alpha=16`, `lora_dropout=0.05`.
- [ ] PEFT targets `["c_attn", "c_proj"]`.
- [ ] PEFT training uses the same `TrainCfg`.
- [ ] Hand-written and PEFT trainable parameter counts are compared.
- [ ] Hand-written and PEFT PPL values are compared.
- [ ] Q4-Q5 are answered from the printed numbers.

Final submission:

- [ ] The notebook runs top to bottom without errors.
- [ ] All required answers Q1-Q5 are filled in.
- [ ] The saved notebook includes outputs, not just code.
- [ ] Any optional bonus work is clearly separated from the required solution.
