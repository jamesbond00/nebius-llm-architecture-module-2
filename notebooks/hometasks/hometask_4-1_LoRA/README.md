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

#### Why Recursive Injection Is Needed

GPT-2 is a nested PyTorch module. The layers we want are not direct children of
the top-level model. They live inside each transformer block:

```text
model.transformer.h[0].attn.c_attn
model.transformer.h[0].attn.c_proj
model.transformer.h[0].mlp.c_proj
```

That means `inject_lora()` needs to walk the module tree recursively. At each
level, it looks at the immediate children:

```python
for name, child in module.named_children():
```

The important detail is that `name` is the local attribute name, such as
`"c_attn"` or `"c_proj"`, not the full dotted path.

This is exactly what we want. Matching by local child name catches both:

```text
attn.c_proj
mlp.c_proj
```

because both are children named `c_proj` under different parent modules.

#### How Replacement Works

When a child module matches the target name and is an `nn.Linear`, replace that
child in its parent:

```python
setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
```

`setattr` is the key operation. It updates the parent module so that, for
example:

```text
block.attn.c_attn
```

now points to a `LoRALinear` wrapper instead of the original `nn.Linear`.

The original linear layer is not discarded. It is stored inside the wrapper as:

```python
self.base = base
```

So the model still has the original projection, plus the new LoRA branch.

#### Why Check `isinstance(child, nn.Linear)`?

The homework first converts GPT-2's `Conv1D` projections to `nn.Linear`. After
that conversion, the intended targets are plain linear layers.

The type check prevents accidentally wrapping a non-linear module that happens
to have a matching name. It also makes the function easier to reason about:

```text
target name + Linear type = safe to wrap
```

#### Complete `inject_lora()` Implementation

The function should replace matching children and recurse into everything else:

```python
def inject_lora(
    module: nn.Module,
    target_names: tuple[str, ...] = ("c_attn", "c_proj"),
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> int:
    """Recursively swap nn.Linear children whose attribute name is in target_names.

    Returns the number of modules wrapped.
    """
    n_wrapped = 0
    for name, child in module.named_children():
        if name in target_names and isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            n_wrapped += 1
        else:
            n_wrapped += inject_lora(
                child,
                target_names=target_names,
                r=r,
                alpha=alpha,
                dropout=dropout,
            )

    return n_wrapped
```

Do not recurse into a child after replacing it. Once `c_attn` or `c_proj` has
been wrapped, that subtree is already handled.

#### Freezing Everything Except LoRA

After injection, freeze every parameter whose name does not contain `"lora_"`:

```python
def freeze_non_lora(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name
```

This works because the new trainable parameters are named:

```text
lora_A
lora_B
```

The frozen base parameters inside each wrapper have names like:

```text
transformer.h.0.attn.c_attn.base.weight
transformer.h.0.attn.c_attn.base.bias
```

Those names do not contain `"lora_"`, so they stay frozen.

This is also why it is cleaner to inject first and freeze second. After all LoRA
modules exist, one pass over `named_parameters()` can set the correct training
state for the whole model.

#### What the Checks Prove

After injection and freezing, the notebook checks:

```python
assert n_wrapped == 36
assert trainable < 1_500_000
assert_same_device(model)
```

These checks protect against different classes of mistakes:

- `n_wrapped == 36`: confirms the recursive traversal found all GPT-2 target
  projections.
- `trainable < 1_500_000`: confirms the base model is frozen and only adapter
  weights are trainable.
- `assert_same_device(model)`: confirms injected LoRA parameters were created
  on the same device as the rest of the model.

If `n_wrapped` is too low, the recursion or target-name matching is wrong. If
the trainable count is too high, freezing is wrong. If the device check fails,
the `LoRALinear` constructor is probably creating tensors on CPU by accident.

#### Common Mistakes

- Matching full paths instead of local child names. In this notebook, match
  `name in ("c_attn", "c_proj")`.
- Forgetting `setattr`; creating a wrapper object is not enough unless it is
  assigned back onto the parent module.
- Freezing before injection and then forgetting to leave new LoRA params
  trainable.
- Checking for `"lora"` too broadly. Use `"lora_"` so the naming convention is
  explicit.
- Forgetting that `c_proj` appears in both attention and MLP blocks. Both are
  intended targets here.

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

#### TODO 7.1: Build the AdamW Optimizer

At this point, `freeze_non_lora(model)` should already have set:

```text
lora_A and lora_B: requires_grad=True
everything else:   requires_grad=False
```

So the optimizer should be built from only the parameters that still require
gradients:

```python
trainable_params = [p for p in model.parameters() if p.requires_grad]

optim = torch.optim.AdamW(
    trainable_params,
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
)
```

This trains the same model behavior as passing `model.parameters()`, because
frozen parameters do not receive gradients. But it is much more memory-efficient.
AdamW stores optimizer state, usually including moving averages, for every
parameter it manages. If you pass all of GPT-2's frozen parameters to AdamW, the
optimizer may allocate unnecessary state for weights that should never update.

For LoRA, that would defeat part of the point of parameter-efficient
fine-tuning. The base GPT-2 weights should participate in the forward pass, but
only the small adapter weights should participate in optimization.

You can quickly confirm the optimizer is scoped correctly by checking the
trainable parameter count before training:

```python
trainable, total = count_params(model)
print(trainable, total)
```

For this homework, `trainable` should be about:

```text
811,008
```

If it is close to the full GPT-2 parameter count, freezing did not work.

### 8. Save and Load the Adapter

Save only the LoRA parameters:

```text
keys containing "lora_"
```

Then load those parameters into a fresh GPT-2 with the same LoRA injection
configuration. The notebook sets both models to eval mode before comparing
logits, because dropout must be disabled for deterministic comparison.

#### TODO 9.1: Save Only LoRA Weights

The adapter checkpoint should not contain GPT-2's frozen base weights. It should
only contain parameters whose names include `"lora_"`:

```python
def save_adapter(model: nn.Module, path: str) -> None:
    adapter_state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if "lora_" in name
    }
    torch.save(adapter_state, path)
```

`param.detach()` removes any autograd graph connection. `.cpu()` makes the file
portable across MPS, CUDA, and CPU runtimes.

This works because the trainable adapter parameters are named:

```text
lora_A
lora_B
```

and the frozen base weights are named through `.base.weight` and `.base.bias`.

#### TODO 9.2: Load Adapter Weights Strictly

To load the adapter, create a dictionary of the fresh model's parameters, then
copy each saved tensor into the matching live parameter:

```python
def load_adapter(model: nn.Module, path: str) -> None:
    adapter_state = torch.load(path, map_location=device)
    params = dict(model.named_parameters())

    for name, value in adapter_state.items():
        assert name in params, f"Saved adapter key {name} not found in model."
        params[name].data.copy_(value.to(device=params[name].device, dtype=params[name].dtype))
```

The strict key check matters. The fresh GPT-2 must be prepared in the same way
as the trained model:

```text
load GPT-2 -> convert Conv1D to Linear -> inject LoRA with same r/alpha/dropout -> load adapter
```

If the LoRA injection config changes, the parameter names or shapes may not
match. It is better to fail loudly than silently load a wrong adapter.

The `.copy_()` call updates the existing `nn.Parameter` in-place. That keeps the
module structure intact and only replaces tensor values.

#### Expected Adapter Size

This homework trains about 811,008 LoRA parameters. In float32, the raw tensor
storage is roughly:

```text
811,008 * 4 bytes = 3,244,032 bytes = about 3.24 MB
```

The observed adapter file size was:

```text
Adapter file size: 3.27 MB
```

The small overhead is normal `torch.save` metadata. This confirms the checkpoint
contains only LoRA adapter weights, not the full GPT-2 model.

The adapter round-trip check passed:

```text
max abs diff: 0.0
OK: adapter round-trips.
```

This means a fresh GPT-2 with the same LoRA injection setup produced identical
logits after loading the saved adapter. The GPT-2 loading report showed
unexpected `attn.bias` keys:

```text
h.{0...11}.attn.bias | UNEXPECTED
```

This did not affect the result. The exact-logit round-trip check is the
important validation signal.

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

## Run Log

### Local MPS Smoke Run

The local Apple Silicon/MPS run proved the hand-written LoRA implementation was
learning, but was slow enough that it was interrupted during epoch 1.

Observed local baseline:

```text
Baseline Shakespeare-val PPL : 93.53
Baseline Control      PPL    : 16.92
```

Observed local training progress:

```text
epoch 0 step    0 | train loss 4.3586 | lr 2.00e-05
epoch 0 step   50 | train loss 4.2260 | lr 2.89e-04
epoch 0 step  100 | train loss 3.8874 | lr 2.43e-04
epoch 0 step  150 | train loss 3.5215 | lr 1.71e-04
== epoch 0 val loss 3.7247 (ppl 41.46) ==
epoch 1 step  200 | train loss 3.9274 | lr 9.39e-05
```

This was enough to validate the implementation locally:

```text
Shakespeare PPL: 93.53 -> 41.46 after epoch 0
```

The local run was then stopped and the full run moved to Colab/CUDA.

### Colab CUDA Hand-Written LoRA Run

The Colab run reproduced the same baseline metrics:

```text
Baseline Shakespeare-val PPL : 93.53
Baseline Control      PPL    : 16.92
```

Final hand-written LoRA training log with:

```python
cfg = TrainCfg(epochs=2, batch_size=8, lr=3e-4)
```

was:

```text
epoch 0 step    0 | train loss 4.4137 | lr 2.00e-05
epoch 0 step   50 | train loss 4.2209 | lr 2.89e-04
epoch 0 step  100 | train loss 3.8991 | lr 2.43e-04
epoch 0 step  150 | train loss 3.5237 | lr 1.71e-04
== epoch 0 val loss 3.7285 (ppl 41.62) ==
epoch 1 step  200 | train loss 3.9718 | lr 9.39e-05
epoch 1 step  250 | train loss 3.6636 | lr 3.17e-05
epoch 1 step  300 | train loss 4.0026 | lr 1.40e-06
== epoch 1 val loss 3.7055 (ppl 40.67) ==
```

Current hand-written LoRA result from the training cell:

```text
Shakespeare-val PPL: 93.53 -> 40.67
```

The biggest improvement happened in epoch 0. Epoch 1 still improved validation
PPL slightly, from 41.62 to 40.67.

The final PPL comparison cell reported:

```text
metric                            before     after     delta
Shakespeare-val PPL                93.53     41.68    -51.85
Control (P&P) PPL                  16.92     18.51     +1.59
```

Use these values for Q3. The Shakespeare validation PPL dropped sharply, while
the control PPL rose slightly. That is the expected specialization trade-off:
the LoRA adapter moved GPT-2 toward Shakespeare-style text, with a small cost on
the unrelated Pride & Prejudice control text.

### Q2-Q3 Answer Notes

The post-training samples showed a clear formatting and style shift:

```text
ROMEO:
How long that time will be, and how it will be, and how it will be.

LUKE ROBERTSON:
Well, what is it,
that you see him, my lord?
```

```text
To be, or not to be, for you?

RICHARD:
It is hard, and it is not at all pleasant to be a man.

MENINIUS:
But, my lord, it is indeed so.
```

```text
Once upon a time in fair Verona,
She beheld the prince, and called him lord.

ROME:
How did he not do it?
```

For Q2, a concise answer can say:

```text
Shift 1 (vocabulary / diction): The outputs shifted toward theatrical,
Shakespeare-like diction. The fine-tuned model uses phrases such as "my lord",
"Wherefore", "prince", "noble", and "glorious", while the baseline drifted into
modern prose, legal language, or generic storytelling.

Shift 2 (formatting / structure): The outputs became much more play-like. They
now use speaker labels such as ROMEO, RICHARD, MENINIUS, ROME, and BRIAN,
followed by short dialogue lines. The baseline mostly produced long modern
paragraphs.

Shift 3 (named entities): The model learned the speaker-label pattern, but not
perfectly. Some names are Shakespeare-like, while others, such as LUKE ROBERTSON
or BRIAN, are odd or modern.
```

For Q3, use:

```text
Shakespeare-val PPL: 93.53 -> 41.68
Control (P&P) PPL:  16.92 -> 18.51
```

`Control (P&P) PPL` means perplexity on the Pride & Prejudice control excerpt.
It is included to check whether specializing toward TinyShakespeare hurts
performance on unrelated prose.

The control PPL moved up by 1.59, while Shakespeare PPL dropped by 51.85. The
control movement is much smaller than the Shakespeare improvement. This means
the LoRA adapter successfully specialized GPT-2 toward Shakespeare style, with a
small regression on unrelated Pride & Prejudice prose.

### Colab CUDA PEFT Comparison Run

The PEFT setup matched the hand-written LoRA configuration:

```text
r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["c_attn", "c_proj"]
```

PEFT reported the same trainable parameter count:

```text
trainable params: 811,008 || all params: 125,250,816 || trainable%: 0.6475
```

It also emitted the expected GPT-2 `Conv1D` warning:

```text
fan_in_fan_out is set to False but the target module is `Conv1D`.
Setting fan_in_fan_out to True.
```

That warning is normal. PEFT applies LoRA directly to GPT-2's original `Conv1D`
modules, while the hand-written implementation converts those modules to
`nn.Linear` first.

PEFT training completed with the same `TrainCfg`:

```text
epoch 0 step    0 | train loss 4.6753 | lr 2.00e-05
epoch 0 step   50 | train loss 3.8433 | lr 2.89e-04
epoch 0 step  100 | train loss 3.8459 | lr 2.43e-04
epoch 0 step  150 | train loss 3.7210 | lr 1.71e-04
== epoch 0 val loss 3.7203 (ppl 41.28) ==
epoch 1 step  200 | train loss 3.8963 | lr 9.39e-05
epoch 1 step  250 | train loss 3.8948 | lr 3.17e-05
epoch 1 step  300 | train loss 3.6499 | lr 1.40e-06
== epoch 1 val loss 3.6982 (ppl 40.38) ==
```

Final quantitative comparison:

```text
metric                           hand-rolled      PEFT
Shakespeare-val PPL                    41.68     41.41
Control (P&P) PPL                      18.51     18.75
```

The Shakespeare validation results are very close: PEFT is lower by only 0.27
PPL, well within 10%. This supports the conclusion that the hand-written LoRA
implementation matches PEFT behavior for this setup.

PEFT generations were also qualitatively similar: they used speaker labels,
short dialogue turns, line breaks, and Shakespeare-like phrases such as "my
lord", "sir", "king", and "prince". Like the hand-written LoRA run, PEFT also
invented some odd speaker names such as `RICHARDIUS`, `MEENESTO`, and `BUDDY`.

### Submission Notebook

The completed submission notebook is:

```text
notebooks/hometasks/hometask_4-1_LoRA/homework_lora_Alex_Colab_20260516.ipynb
```

It includes the completed Q1-Q5 workflow, visible run outputs, adapter
round-trip output, and PEFT comparison. The optional Q6 bonus was not attempted.

## Validation Checklist

Before training:

- [x] `LoRALinear` matches the base layer at initialization.
- [x] Perturbing `lora_B` changes the output.
- [x] `merged_weight()` matches the forward pass.
- [x] LoRA injection wraps exactly 36 modules.
- [x] Trainable params are around 811,008.
- [x] Trainable fraction is about 0.65%.
- [x] All model parameters are on one device.

After hand-written LoRA training:

- [x] Baseline generations are visible.
- [x] Post-training generations are visible.
- [x] Shakespeare validation PPL before/after is recorded.
- [x] Pride & Prejudice control PPL before/after is recorded.
- [x] Q1-Q3 are answered using actual outputs and metrics.

Adapter check:

- [x] Adapter file is saved.
- [x] Adapter file size is printed.
- [x] Fresh GPT-2 plus loaded adapter reproduces trained-model logits.

PEFT comparison:

- [x] PEFT config uses `r=8`, `lora_alpha=16`, `lora_dropout=0.05`.
- [x] PEFT targets `["c_attn", "c_proj"]`.
- [x] PEFT training uses the same `TrainCfg`.
- [x] Hand-written and PEFT trainable parameter counts are compared.
- [x] Hand-written and PEFT PPL values are compared.
- [x] Q4-Q5 are answered from the printed numbers.

Final submission:

- [x] The notebook runs top to bottom without errors.
- [x] All required answers Q1-Q5 are filled in.
- [x] The saved notebook includes outputs, not just code.
- [x] Optional Q6 bonus was not attempted.
