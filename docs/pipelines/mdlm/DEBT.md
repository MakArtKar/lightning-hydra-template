# MDLM Implementation Technical Debt

This document outlines the technical debt in the current Lightning-Hydra MDLM implementation compared to the official [MDLM repository](https://github.com/kuleshov-group/mdlm/tree/master).

**Note**: This is not a strict replication guide. We implement MDLM following our Lightning-Hydra architecture principles while maintaining algorithmic correctness.

## Executive Summary

**Current State**: Simplified masked language modeling with basic random masking.

**Target State**: Complete MDLM implementation with absorbing-state D3PM diffusion, proper loss masking, and inference capabilities.

______________________________________________________________________

## Scope: Datasets and Model Architectures

This implementation focuses on:

### Datasets

- **Primary**: LM1B (One Billion Word Benchmark)
- **Future**: OpenWebText, text8 (lower priority)

### Model Architectures

- **Primary**: DiT-based transformer (~110M parameters)
- **Alternative**: x_transformers (current - acceptable with proper configuration)
- **Future**: Mamba-based models (low priority)

______________________________________________________________________

## CRITICAL TECHNICAL DEBT (Must Fix)

### ✅ DEBT-1: Loss Computation on Masked Positions Only \[IMPLEMENTED\]

**Priority**: VERY CRITICAL - Fix First

**Status**: ✅ COMPLETED (2025-11-05)

**Implementation**:

Created `MaskLoss` class in `ml_core/transforms/diffusion/mdlm.py`:

- Takes `ignore_index: int = -100` parameter in `__init__`
- `__call__` takes `target` and `mask` parameters
- Sets non-masked positions (where `mask = False`) to `ignore_index`
- This ensures CrossEntropyLoss only computes on masked positions

**Config Update**:

```yaml
# In configs/model/diffusion/mdlm.yaml
forward_fn:
  mask_loss:
    _target_: ml_core.transforms.base.WrapTransform
    transform:
      _target_: ml_core.transforms.diffusion.mdlm.MaskLoss
    mapping:
      input_ids: target
      mask: mask
    new_key: masked_target

criterions:
  mapping:
    ce:
      reshaped_logits: input
      masked_target: target  # Loss only on masked positions
```

**Action Items**:

1. [x] In `ml_core/transforms/diffusion/mdlm.py` create `MaskLoss` class
2. [x] Create unit tests for it in `tests/test_transforms/test_functions/test_diffusion/test_mdlm/test_mask_loss.py`
3. [x] Update `forward_fn` in `configs/model/diffusion/mdlm.yaml`

**Files Modified**:

- `ml_core/transforms/diffusion/mdlm.py` - Added MaskLoss class
- `tests/test_transforms/test_functions/test_diffusion/test_mdlm/test_mask_loss.py` - 11 comprehensive unit tests
- `configs/model/diffusion/mdlm.yaml` - Updated forward_fn to use MaskLoss

**Test Results**: All tests passing ✅

______________________________________________________________________

### 🔴 DEBT-2: Model Size Alignment (110M Parameters)

**Priority**: CRITICAL

**Current State**:

```yaml
params:
  model:
    dim: 16        # Toy size
    depth: 2       # Toy size
    heads: 4       # Toy size
```

**Required for LM1B DiT**:

```yaml
params:
  model:
    dim: 768
    depth: 12
    heads: 12
    num_tokens: 30522  # BERT vocab
```

**Validation**:

- Expected parameters: ~110M
- Formula: Check with `sum(p.numel() for p in model.parameters())`

**Action Items**:

1. [ ] Update `configs/experiment/diffusion/mdlm.yaml` with production parameters
2. [ ] Verify parameter count matches ~110M

**Files to Modify**:

- `configs/experiment/diffusion/mdlm.yaml`

______________________________________________________________________

### 🔴 DEBT-3: Sampling/Inference Implementation

**Priority**: CRITICAL

**Current State**: No sampling capabilities - training only.

**Required**: At minimum, implement basic DDPM sampling for inference.

**Implementation**:

Create `ml_core/models/components/diffusion/samplers/ddpm_sampler.py`:

```python
class DDPMSampler:
    """Basic ancestral sampling for MDLM."""

    def __init__(self, model, tokenizer, seq_length, num_steps=1000):
        self.model = model
        self.tokenizer = tokenizer
        self.num_steps = num_steps
        self.seq_length = seq_length

    def sample(self, batch_size, device):
        """Generate samples starting from all mask tokens."""
        # Start with all [MASK]
        x = torch.full((batch_size, self.seq_length),
                      self.tokenizer.mask_token_id,
                      device=device)

        # Reverse diffusion process
        for t in reversed(range(self.num_steps)):
            # Get model predictions
            with torch.no_grad():
                logits = self.model(x)

            # Sample from logits
            probs = F.softmax(logits, dim=-1)
            x_new = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
            x_new = x_new.view(batch_size, self.seq_length)

            # Determine which tokens to update based on diffusion schedule
            update_mask = self._get_update_mask(t, batch_size, self.seq_length, device)
            x = torch.where(update_mask, x_new, x)

        return x

    def _get_update_mask(self, t, batch_size, device):
        """Determine which positions to update at step t."""
        # Fraction of tokens still masked at step t
        frac_masked = t / self.num_steps
        num_masked = int(frac_masked * self.seq_length)

        # Randomly select positions to keep masked
        mask = torch.zeros(batch_size, self.seq_length, device=device).bool()
        for b in range(batch_size):
            indices = torch.randperm(self.seq_length, device=device)[:num_masked]
            mask[b, indices] = True

        return ~mask  # Return positions to update
```

**Action Items**:

1. [ ] Implement `DDPMSampler` class
2. [ ] Create unit tests in `tests/test_model/test_components/test_samplers/test_ddpm_sampler.py` to test sampler
3. [ ] Implement `SamplingCallback` that takes sampler in `__init__` and on validation and testing epochs end samples
4. [ ] Add callback configuration in experiment config
5. [ ] Test sampling produces valid sequences
6. [ ] Add temperature parameter for controlled generation
7. [ ] Add `num_steps` parameter that controls number of diffusion steps

**Files to Create**:

- `ml_core/models/components/samplers/__init__.py`
- `ml_core/models/components/samplers/ddpm_sampler.py`

______________________________________________________________________

### 🔴 DEBT-4: Optimizer Configuration (AdamW)

**Priority**: CRITICAL

**Current State**:

```yaml
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001
  weight_decay: 0.0
```

**Required State**:

```yaml
optimizer:
  _target_: torch.optim.AdamW
  _partial_: true
  lr: 2.0e-4
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1.0e-8
```

**Action Items**:

1. [ ] Update optimizer in model config
2. [ ] Verify weight decay is properly applied

**Files to Modify**:

- `configs/model/diffusion/mdlm.yaml`

______________________________________________________________________

### 🔴 DEBT-5: Learning Rate Schedule (Cosine with Warmup)

**Priority**: CRITICAL

**Current State**:

```yaml
scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
```

**Required State**:

```yaml
scheduler:
  _target_: transformers.get_cosine_schedule_with_warmup
  _partial_: true
  num_warmup_steps: 1000
  num_training_steps: 1000000
```

**Action Items**:

1. [ ] Add transformers to requirements if not present
2. [ ] Update scheduler configuration
3. [ ] Modify `BaseLitModule.configure_optimizers()` to return scheduler with `interval="step"`
4. [ ] Verify LR schedule is logged correctly

**Files to Modify**:

- `configs/model/diffusion/mdlm.yaml`
- `ml_core/models/base_module.py` (scheduler interval configuration)
- `requirements.txt` (ensure transformers is included)

______________________________________________________________________

### 🔴 DEBT-6: Training Hyperparameters

**Priority**: CRITICAL

**Current State**:

```yaml
trainer:
  min_epochs: 5
  max_epochs: 5
  gradient_clip_val: 0.5

data:
  batch_size: 128
```

**Required State**:

```yaml
trainer:
  max_steps: 1000000
  gradient_clip_val: 1.0
  val_check_interval: 10000
  log_every_n_steps: 100

data:
  batch_size: 512  # Or use gradient accumulation

trainer:
  accumulate_grad_batches: 4  # If GPU memory limited
```

**Action Items**:

1. [ ] Change from epoch-based to step-based training
2. [ ] Update gradient clipping value
3. [ ] Configure batch size with gradient accumulation if needed
4. [ ] Set validation and logging intervals

**Files to Modify**:

- `configs/experiment/diffusion/mdlm.yaml`
- `configs/trainer/default.yaml` (or create `configs/trainer/mdlm.yaml`)

______________________________________________________________________

### 🔴 DEBT-7: EMA (Exponential Moving Average)

**Priority**: CRITICAL

**Current State**: No EMA implementation.

**Required**: EMA of model parameters improves generation quality.

**Implementation**:

Option 1 - Use Lightning callback:

```python
# ml_core/callbacks/ema.py
from lightning.pytorch.callbacks import EMA

# In config
callbacks:
  ema:
    _target_: lightning.pytorch.callbacks.EMA
    decay: 0.9999
    apply_ema_every_n_steps: 1
```

Option 2 - Manual implementation:

```python
class EMACallback(Callback):
    def __init__(self, decay=0.9999):
        self.decay = decay
        self.ema_params = {}

    def on_train_start(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                self.ema_params[name] = param.data.clone()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    self.ema_params[name].mul_(self.decay).add_(
                        param.data, alpha=1 - self.decay
                    )
```

**Action Items**:

1. [ ] Implement EMA callback or use PyTorch Lightning's built-in EMA
2. [ ] Add to experiment config callbacks
3. [ ] Use EMA weights for validation/sampling
4. [ ] Save EMA weights in checkpoints

**Files to Create/Modify**:

- `ml_core/callbacks/ema.py` (if custom implementation)
- `configs/callbacks/ema.yaml`
- `configs/experiment/diffusion/mdlm.yaml`

______________________________________________________________________

### 🔴 DEBT-8: Mixed Precision (bfloat16)

**Priority**: CRITICAL

**Current State**: No precision specified (defaults to fp32).

**Required State**:

```yaml
trainer:
  precision: bf16-mixed  # or "16-mixed" for older GPUs
```

**Benefits**:

- 2x memory reduction
- 1.5-2x speedup
- Better numerical stability than fp16 for transformers

**Action Items**:

1. [ ] Enable bf16 precision in trainer config
2. [ ] Verify GPU supports bfloat16 (A100, H100, or Ampere+)
3. [ ] Test training stability
4. [ ] Fall back to fp16 or fp32 if needed

**Files to Modify**:

- `configs/trainer/default.yaml` or `configs/experiment/diffusion/mdlm.yaml`

______________________________________________________________________

## HIGH PRIORITY TECHNICAL DEBT

### 🟡 DEBT-9: Perplexity Metric During Validation/Testing

**Priority**: HIGH

**Current State**: Only loss logging.

**Required**: Calculate perplexity on validation/test sets.

**Implementation**:

```yaml
callbacks:
  metrics_callback:
    metrics:
      _target_: ml_core.models.utils.MetricsComposition
      metrics:
        perplexity:
          _target_: torchmetrics.text.Perplexity
          ignore_index: -100
      mapping:
        perplexity:
          preds: logits
          target: input_ids
```

**Action Items**:

1. [ ] Add perplexity metric to experiment config
2. [ ] Configure ignore_index for padding tokens
3. [ ] Ensure metric only computes on masked positions
4. [ ] Log perplexity alongside loss

**Files to Modify**:

- `configs/experiment/diffusion/mdlm.yaml`

______________________________________________________________________

### 🟡 DEBT-10: Logging Enhancements

**Priority**: HIGH

**Current State**: Basic loss logging, wandb available with `logger=wandb`.

**Required Additions**:

- Learning rate logging
- Gradient norm logging
- Sample quality tracking
- Model checkpoint monitoring

**Implementation**:

In `BaseLitModule`, add to `training_step`:

```python
def training_step(self, batch, batch_idx):
    result = self.model_step(batch, "train")

    # Log learning rate
    lr = self.optimizers().param_groups[0]['lr']
    self.log("train/lr", lr, on_step=True, on_epoch=False)

    return result

def on_before_optimizer_step(self, optimizer):
    # Log gradient norms
    grad_norm = torch.nn.utils.clip_grad_norm_(
        self.parameters(),
        self.trainer.gradient_clip_val or float('inf')
    )
    self.log("train/grad_norm", grad_norm)
```

**Action Items**:

1. [ ] Add LR logging to training step
2. [ ] Add gradient norm logging
3. [ ] Configure W&B with proper tags/group
4. [ ] Add sample generation callback (log samples every N steps)

**Files to Modify**:

- `ml_core/models/base_module.py`
- `configs/experiment/diffusion/mdlm.yaml` (wandb config)

______________________________________________________________________

### 🟡 DEBT-11: Model Checkpointing Configuration

**Priority**: HIGH

**Current State**: Default checkpoint behavior.

**Required State**:

```yaml
callbacks:
  model_checkpoint:
    monitor: val/loss
    mode: min
    save_top_k: 3
    save_last: True
    every_n_train_steps: 10000
    filename: "mdlm-{step:07d}-{val/loss:.4f}"
```

**Action Items**:

1. [ ] Configure checkpoint callback
2. [ ] Save checkpoints every 10k steps
3. [ ] Keep top-3 and last checkpoint
4. [ ] Include EMA weights in checkpoints

**Files to Modify**:

- `configs/callbacks/model_checkpoint.yaml`
- `configs/experiment/diffusion/mdlm.yaml`

______________________________________________________________________

## LOW PRIORITY TECHNICAL DEBT

### 🟢 DEBT-12: Advanced Noise Schedules

**Priority**: LOW

**Current State**: Linear noise schedule (k ~ U\[1, L\]) is implemented.

**Optional Additions**:

- Cosine noise schedule
- Log-linear noise schedule
- Configurable schedule selection

**Note**: Current linear implementation is sufficient. These are optimizations, not critical requirements.

**Action Items** (Optional):

1. [ ] Create `ml_core/transforms/diffusion/noise_schedule.py`
2. [ ] Implement cosine and log-linear schedules
3. [ ] Make schedule configurable in experiment config

______________________________________________________________________

### 🟢 DEBT-13: Advanced Sampling Methods

**Priority**: LOW

**Current State**: None (DEBT-3 covers basic sampling).

**Optional Additions**:

- `ddpm_cache`: Efficient MDLM sampler (~3-4x faster)
- `analytic`: SEDD-style analytic sampler
- Temperature and top-k/top-p sampling controls

**Note**: Basic DDPM sampling (DEBT-3) is sufficient for initial implementation.

**Action Items** (Optional):

1. [ ] Implement `DDPMCacheSampler` for faster inference
2. [ ] Implement `AnalyticSampler` for alternative sampling
3. [ ] Add temperature and nucleus sampling controls

______________________________________________________________________

### 🟢 DEBT-14: Generative Perplexity Evaluation

**Priority**: LOW (Separate from DEBT-9)

**Current State**: No generative evaluation.

**Optional**: Evaluate generated samples under pre-trained GPT-2.

**Implementation**:

```python
class GenerativePerplexityCallback(Callback):
    def __init__(self, gpt2_model, num_samples=100):
        self.gpt2 = gpt2_model
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        # Generate samples
        samples = self.generate_samples(pl_module, self.num_samples)

        # Evaluate under GPT-2
        ppl = self.compute_perplexity(samples)

        trainer.logger.log_metrics({"val/gen_ppl": ppl})
```

**Action Items** (Optional):

1. [ ] Implement generative perplexity callback
2. [ ] Load pre-trained GPT-2 for evaluation
3. [ ] Generate samples during validation
4. [ ] Log generative perplexity

______________________________________________________________________

### 🟢 DEBT-15: Time Conditioning in Model

**Priority**: LOW

**Current State**: No time conditioning.

**Optional**: Add time embeddings to model for full diffusion framework.

**Note**: MDLM can work without explicit time conditioning using the masking ratio as implicit time signal. This is an enhancement, not a requirement.

**Implementation** (if needed):

```python
class TimeConditioning(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, t):
        # t: [B] - time steps
        time_emb = self.get_timestep_embedding(t, x.shape[-1])
        time_emb = self.time_mlp(time_emb)
        return x + time_emb.unsqueeze(1)
```

**Action Items** (Optional):

1. [ ] Add time embedding module to model
2. [ ] Modify forward pass to include time step
3. [ ] Update masking to use time-based schedule

______________________________________________________________________

### 🟢 DEBT-16: Alternative Model Sizes

**Priority**: LOW (Separate from DEBT-2)

**Current State**: Only toy model (16/2/4).

**Optional Configurations**:

- Small: dim=512, depth=6, heads=8 (~30M params)
- Medium: dim=768, depth=12, heads=12 (~110M params) ✓ DEBT-2
- Large: dim=1024, depth=24, heads=16 (~350M params)

**Action Items** (Optional):

1. [ ] Create config variants for different sizes
2. [ ] Document parameter counts for each
3. [ ] Test memory requirements

______________________________________________________________________

### 🟢 DEBT-17: Data Filtering and Preprocessing

**Priority**: LOW (Separate from DEBT-6)

**Current State**: Basic tokenization with padding.

**Optional Enhancements**:

- Filter sequences by length (min/max)
- Remove empty sequences
- Dynamic batching by length
- Data quality filtering

**Action Items** (Optional):

1. [ ] Add length filtering to data config
2. [ ] Implement quality filters if needed
3. [ ] Add dynamic batching for efficiency

______________________________________________________________________

### 🟢 DEBT-18: Additional Datasets

**Priority**: LOW

**Current State**: LM1B only.

**Optional Datasets**:

- OpenWebText
- text8
- WikiText-103

**Action Items** (Optional):

1. [ ] Create config files for additional datasets
2. [ ] Verify tokenizer compatibility
3. [ ] Document dataset-specific requirements

______________________________________________________________________

### 🟢 DEBT-19: Alternative Architectures

**Priority**: LOW

**Current State**: x_transformers-based encoder.

**Optional Architectures**:

- Custom DiT implementation (closer to official)
- Mamba-based models
- Hybrid architectures

**Action Items** (Optional):

1. [ ] Implement custom DiT if performance critical
2. [ ] Test Mamba for efficiency comparisons
3. [ ] Document architecture tradeoffs

______________________________________________________________________

## DOCUMENTATION DEBT

### 📝 DEBT-20: Pipeline Documentation

**Priority**: MEDIUM

**Action Items**:

1. [ ] Create `docs/pipelines/mdlm/README.md` with overview
2. [ ] Create `docs/pipelines/mdlm/TRAINING.md` with training guide
3. [ ] Create `docs/pipelines/mdlm/SAMPLING.md` with inference guide
4. [ ] Add example commands and expected results

______________________________________________________________________

### 📝 DEBT-21: Code Documentation

**Priority**: MEDIUM

**Action Items**:

1. [ ] Add docstrings to all MDLM transforms
2. [ ] Document sampling methods
3. [ ] Add type hints throughout
4. [ ] Create docstring examples

______________________________________________________________________

## TESTING DEBT

### 🧪 DEBT-22: Unit Tests

**Priority**: HIGH

**Action Items**:

1. [ ] Test masked loss computation
2. [ ] Test masking logic with different ratios
3. [ ] Test sampling generates valid sequences
4. [ ] Test EMA callback functionality
5. [ ] Test training with bf16 precision

______________________________________________________________________

### 🧪 DEBT-23: Integration Tests

**Priority**: MEDIUM

**Action Items**:

1. [ ] End-to-end training test (fast_dev_run)
2. [ ] Sampling integration test
3. [ ] Checkpoint loading and resuming test
4. [ ] Multi-GPU training test (if available)

______________________________________________________________________

## NOT NEEDED (Clarifications)

### ❌ NOT-1: Multiple Training Modes

**Status**: NOT NEEDED

**Reasoning**: We use separate scripts following architecture:

- `ml_core/train.py` - Training
- `ml_core/eval.py` - Evaluation with checkpoint

No need for `mode=train`, `mode=ppl_eval`, `mode=sample_eval` in single script.

______________________________________________________________________

### ❌ NOT-2: Dataset Split Configuration

**Status**: ALREADY CORRECT

**Current Implementation**: `BaseDataModule` correctly uses:

- `val_ratio` for creating validation split from training data
- Official test split from dataset

This matches the architecture and is correct. No changes needed.

______________________________________________________________________

### ❌ NOT-3: Exact File Structure Replication

**Status**: NOT NEEDED

**Reasoning**: We follow Lightning-Hydra architecture, not original MDLM structure. Our organization:

- Better separation of concerns
- More modular and reusable
- Follows PyTorch Lightning best practices

______________________________________________________________________

## IMPLEMENTATION PRIORITY ORDER

### Phase 1: Core Correctness (Week 1)

1. ✅ **DEBT-1**: Loss on masked positions only ⚠️ MOST CRITICAL \[COMPLETED\]
2. **DEBT-4**: AdamW optimizer
3. **DEBT-5**: Cosine LR schedule with warmup
4. **DEBT-6**: Training hyperparameters (step-based, gradient clip)
5. **DEBT-8**: bf16 precision

### Phase 2: Model and Training (Week 2)

06. **DEBT-2**: Scale model to 110M parameters
07. **DEBT-7**: EMA implementation
08. **DEBT-9**: Perplexity metric
09. **DEBT-10**: Logging enhancements
10. **DEBT-22**: Unit tests for critical components

### Phase 3: Inference (Week 3)

11. **DEBT-3**: Basic DDPM sampling
12. **DEBT-11**: Model checkpointing
13. **DEBT-23**: Integration tests
14. **DEBT-20**: Pipeline documentation

### Phase 4: Optimizations (Optional, Week 4+)

15. **DEBT-12**: Advanced noise schedules
16. **DEBT-13**: Advanced sampling methods
17. **DEBT-14**: Generative perplexity
18. **DEBT-15-19**: Alternative configs and datasets

______________________________________________________________________

## PARAMETER COMPARISON

| Component         | Current             | Required        | Priority   | Status    |
| ----------------- | ------------------- | --------------- | ---------- | --------- |
| **Loss**          |                     |                 |            |           |
| Loss positions    | Masked only         | Masked only     | 🔴 CRITICAL | ✅ DEBT-1  |
| Ignore padding    | Yes                 | Yes             | 🔴 CRITICAL | ✅ DEBT-1  |
| **Model**         |                     |                 |            |           |
| Hidden dim        | 16                  | 768             | 🔴 CRITICAL | ❌ DEBT-2  |
| Depth             | 2                   | 12              | 🔴 CRITICAL | ❌ DEBT-2  |
| Heads             | 4                   | 12              | 🔴 CRITICAL | ❌ DEBT-2  |
| Parameters        | ~5K                 | ~110M           | 🔴 CRITICAL | ❌ DEBT-2  |
| Time conditioning | No                  | No (optional)   | 🟢 LOW      | ✅ OK      |
| **Training**      |                     |                 |            |           |
| Duration          | 5 epochs            | 1M steps        | 🔴 CRITICAL | ❌ DEBT-7  |
| Batch size        | 128                 | 512 (or 128×4)  | 🔴 CRITICAL | ❌ DEBT-7  |
| Optimizer         | Adam                | AdamW           | 🔴 CRITICAL | ❌ DEBT-5  |
| Learning rate     | 1e-3                | 2e-4            | 🔴 CRITICAL | ❌ DEBT-5  |
| Weight decay      | 0.0                 | 0.01            | 🔴 CRITICAL | ❌ DEBT-5  |
| LR schedule       | ReduceLROnPlateau   | Cosine+Warmup   | 🔴 CRITICAL | ❌ DEBT-6  |
| Warmup steps      | 0                   | 1000            | 🔴 CRITICAL | ❌ DEBT-6  |
| Gradient clip     | 0.5                 | 1.0             | 🔴 CRITICAL | ❌ DEBT-7  |
| Precision         | fp32                | bf16            | 🔴 CRITICAL | ❌ DEBT-9  |
| EMA               | No                  | Yes (0.9999)    | 🔴 CRITICAL | ❌ DEBT-8  |
| **Masking**       |                     |                 |            |           |
| Method            | Linear (k~U\[1,L\]) | Same            | ✅ OK       | ✅ OK      |
| Noise schedule    | Linear              | Optional cosine | 🟢 LOW      | ✅ DEBT-12 |
| **Inference**     |                     |                 |            |           |
| Sampling          | None                | DDPM (basic)    | 🔴 CRITICAL | ❌ DEBT-3  |
| Advanced samplers | None                | Optional        | 🟢 LOW      | ⚪ DEBT-14 |
| **Evaluation**    |                     |                 |            |           |
| Perplexity        | No                  | Yes             | 🟡 HIGH     | ❌ DEBT-11 |
| Gen. perplexity   | No                  | Optional        | 🟢 LOW      | ⚪ DEBT-15 |
| **Data**          |                     |                 |            |           |
| Dataset           | lm1b                | lm1b            | ✅ OK       | ✅ OK      |
| Val split         | val_ratio (correct) | Same            | ✅ OK       | ✅ OK      |
| Test split        | Official            | Official        | ✅ OK       | ✅ OK      |
| Filtering         | None                | Optional        | 🟢 LOW      | ⚪ DEBT-18 |

**Legend**:

- 🔴 CRITICAL: Must fix for correct implementation
- 🟡 HIGH: Important for good results
- 🟢 LOW: Nice to have, not critical
- ✅ OK: Already correct
- ❌ Needs fixing
- ⚪ Optional enhancement

______________________________________________________________________

## ESTIMATED EFFORT

### Critical Path (Minimum Viable MDLM)

- ✅ **DEBT-1 (Loss masking)**: 1-2 days \[COMPLETED\]
- **DEBT-4 (AdamW)**: 0.5 days
- **DEBT-5 (LR schedule)**: 0.5 days
- **DEBT-6 (Training params)**: 0.5 days
- **DEBT-2 (Model size)**: 0.5 days
- **DEBT-7 (EMA)**: 1 day
- **DEBT-8 (bf16)**: 0.5 days
- **DEBT-3 (Sampling)**: 2-3 days

**Total Critical Path**: 5-7 days (remaining)

### Full Implementation

- Critical Path: 7-9 days
- High Priority: +3-4 days
- Testing & Docs: +2-3 days
- **Total**: 12-16 days

______________________________________________________________________

## VALIDATION CHECKLIST

Once critical debt is resolved, verify:

- [x] Loss computed only on masked positions
- [ ] Model has ~110M parameters
- [ ] Training runs for 1M steps
- [ ] Effective batch size = 512
- [ ] AdamW with lr=2e-4, wd=0.01
- [ ] Cosine schedule with 1k warmup
- [ ] bf16 precision enabled
- [ ] EMA with decay=0.9999
- [ ] Can generate samples with DDPM
- [ ] Perplexity logged during validation
- [ ] Training converges (loss decreases)
- [ ] Generated samples are coherent

______________________________________________________________________

## REFERENCES

- **Official MDLM Repository**: https://github.com/kuleshov-group/mdlm/tree/master
- **MDLM Paper**: Simple and Effective Masked Diffusion Language Models (NeurIPS 2024)
- **Paper Link**: https://openreview.net/forum?id=L4uaAR4ArM
- **Website**: https://s-sahoo.com/mdlm/
- **HuggingFace Checkpoint**: https://huggingface.co/kuleshov-group/mdlm-owt

### Key Reference Files in Official Repo:

- `main.py` - Training entry point
- `diffusion.py` - Diffusion process (for reference, we implement simplified)
- `noise_schedule.py` - Noise schedules (linear is sufficient)
- `dataloader.py` - Data loading patterns
- `models/dit.py` - DiT architecture (we use x_transformers)

______________________________________________________________________

## NOTES

1. **Architecture Philosophy**: We follow Lightning-Hydra design principles, not exact code structure from official repo.

2. **Loss Masking is Critical**: The most important difference is computing loss only on masked positions (DEBT-4). Fix this first.

3. **Linear Masking is Sufficient**: Current k~U\[1,L\] implementation is algorithmically correct. Cosine schedules are optional optimizations.

4. **Model Size Matters**: Scaling to 110M parameters is critical for comparable results.

5. **EMA and bf16 are Essential**: Both significantly impact training efficiency and quality.

6. **Sampling is Core Functionality**: Basic DDPM sampling must be implemented. Advanced samplers are optional.

7. **Time Conditioning is Optional**: MDLM works with masking ratio as implicit time signal.

8. **Follow Our Architecture**: Use `train.py` and `eval.py` pattern, not multiple modes in one script.

______________________________________________________________________

**Document Version**: 2.0
**Last Updated**: 2025-11-05
**Status**: Active technical debt tracking
**Next Review**: After Phase 1 completion
