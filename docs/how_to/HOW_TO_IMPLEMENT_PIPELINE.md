# How to Implement a Training Pipeline

This guide explains how to implement a complete training pipeline using the Lightning-Hydra template. We use a Masked Discrete Language Model (MDLM) as a concrete example, but the patterns apply to any deep learning task.

## Pipeline Overview

A complete pipeline consists of:

1. **Data Module** - Loading and preprocessing data
2. **Data Transforms** - Preprocessing and augmentation pipeline
3. **Model Architecture** - Neural network components
4. **Forward Function** - Composable data flow through model
5. **Experiment Config** - Hyperparameters and component selection

These components are modular and reusable across different projects.

## Step 1: Data Module Configuration

Configure data loading using `BaseDataModule`, which works with HuggingFace datasets or any PyTorch dataset.

**Example**: Text dataset for language modeling (`configs/data/lm1b.yaml`)

```yaml
defaults:
  - /data/transform@transform.tokenizer: tokenizer

_target_: ml_core.data.base_datamodule.BaseDataModule

hf_dict_dataset:
  _target_: datasets.load_dataset
  _args_:
    - dvruette/lm1b  # Dataset name from HuggingFace

val_ratio: 0.1  # Split ratio for validation

transform:
  _target_: ml_core.transforms.base.ComposeTransform

  # Convert attention mask to boolean for x_transformers
  convert_mask_to_bool:
    _target_: ml_core.transforms.base.WrapTransform
    transform:
      _target_: ml_core.transforms.diffusion.mdlm.ToBool
    mapping:
      attention_mask: tensor
    new_key: attention_mask

batch_size: 128
num_workers: 8
persistent_workers: True
pin_memory: True
```

**Key Concepts**:

- **Defaults composition**: Import reusable transform configs with `@` syntax
- **Automatic splits**: `BaseDataModule` handles train/val/test partitioning
- **Transform chaining**: `ComposeTransform` applies multiple transforms sequentially
- **Key-based routing**: `WrapTransform` applies transforms to specific batch dictionary keys

## Step 2: Data Transform Configuration

Create reusable, composable transform configurations for any preprocessing task.

**Example**: Tokenizer transform for text data (`configs/data/transform/tokenizer.yaml`)

```yaml
_target_: ml_core.transforms.base.WrapTransform

# Tokenizer is provided via params
transform: ${params.data.tokenizer}

mapping:
  text: text

transform_kwargs:
  max_length: ${params.data.max_length}
  padding: max_length
  truncation: True
  return_tensors: pt
```

**Key Concepts**:

- **Parameter interpolation**: `${params.*}` references allow experiment-level configuration
- **Key mapping**: `mapping` routes batch keys to transform arguments
- **Additional arguments**: `transform_kwargs` provides extra parameters to the transform callable

## Step 3: Model Architecture

Define reusable neural network components that can be composed into different architectures.

**Example**: Transformer encoder with embeddings (`configs/model/transformer/encoder.yaml`)

```yaml
_target_: x_transformers.TransformerWrapper
num_tokens: ${params.model.num_tokens}
max_seq_len: ${params.data.max_length}

attn_layers:
  _target_: x_transformers.Encoder
  dim: ${params.model.dim}
  depth: ${params.model.depth}
  heads: ${params.model.heads}
```

**Key Concepts**:

- **Hierarchical composition**: Nest model components (e.g., encoder inside wrapper)
- **Shared parameters**: Use `${params.*}` for hyperparameters shared across components
- **Modularity**: Architecture components can be swapped independently

## Step 4: Forward Function Composition

Compose the complete forward pass by chaining data preprocessing, model inference, and post-processing steps.

**Example**: Masked language model forward function (`configs/model/diffusion/mdlm.yaml`)

```yaml
defaults:
  - _self_  # Load current config first
  - /model/transformer@forward_fn.network.transform: encoder  # Import encoder

_target_: ml_core.models.base_module.BaseLitModule

forward_fn:
  _target_: ml_core.transforms.base.ComposeTransform

  # Step 1: Sample random mask
  sample_mask:
    _target_: ml_core.transforms.base.WrapTransform
    transform:
      _target_: ml_core.transforms.diffusion.mdlm.SampleMask
    mapping:
      input_ids: input_ids
    new_key: mask

  # Step 2: Apply mask to input_ids
  masked_input_ids:
    _target_: ml_core.transforms.base.WrapTransform
    transform:
      _target_: ml_core.transforms.diffusion.mdlm.MaskInputIds
      tokenizer: ${params.data.tokenizer}
    mapping:
      input_ids: input_ids
      mask: mask
    new_key: masked_input_ids

  # Step 3: Forward through network (encoder imported via defaults)
  network:
    _target_: ml_core.transforms.base.WrapTransform
    transform: ???  # Filled by defaults
    mapping:
      masked_input_ids: x
      attention_mask: mask
    new_key: logits

  # Step 4: Reshape for loss computation
  reshape_logits:
    _target_: ml_core.transforms.base.WrapTransform
    transform:
      _target_: torch.transpose
      _partial_: true
      dim0: -1
      dim1: -2
    mapping:
      logits: x
    new_key: reshaped_logits

criterions:
  _target_: ml_core.models.utils.CriterionsComposition
  criterions:
    ce:
      _target_: torch.nn.CrossEntropyLoss
  weights:
    ce: 1.0
  mapping:
    ce:
      reshaped_logits: input  # batch_key: criterion_arg
      input_ids: target

optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.001

scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  _partial_: true
  mode: min
  factor: 0.1
  patience: 10

compile: false
```

### Understanding Defaults with `@` Syntax

The line:

```yaml
- /model/transformer@forward_fn.network.transform: encoder
```

Means:

- **Load**: `configs/model/transformer/encoder.yaml`
- **Place at**: `forward_fn.network.transform` (instead of root)
- **Use**: `encoder` as the config name

This fills the `transform: ???` placeholder in the `network` step.

**Key Concepts**:

- **Execution order**: `_self_` controls when current config is merged with defaults
- **Sequential processing**: `ComposeTransform` applies transforms in definition order
- **Dictionary accumulation**: Each transform receives full batch dict and adds new keys
- **Flexible routing**: `mapping` defines which batch keys feed into each transform
- **Lazy binding**: `???` creates placeholders filled by defaults or overrides

## Step 5: Experiment Configuration

Bind all components together and specify hyperparameters for a specific experiment.

**Example**: Language model training experiment (`configs/experiment/diffusion/mdlm.yaml`)

```yaml
# @package _global_

defaults:
  - override /data: lm1b
  - override /model: diffusion/mdlm
  - override /callbacks: default
  - override /trainer: default

tags: ["lm1b", "mdlm"]

seed: 12345

trainer:
  min_epochs: 5
  max_epochs: 5
  gradient_clip_val: 0.5

params:
  model:
    dim: 16
    depth: 2
    heads: 4
    num_tokens: 30522  # BERT vocab size
  data:
    tokenizer:
      _target_: transformers.AutoTokenizer.from_pretrained
      pretrained_model_name_or_path: "bert-base-uncased"
    max_length: 128

logger:
  wandb:
    tags: ${tags}
    group: "mdlm"
```

**Key Concepts**:

- **Global scope**: `@package _global_` merges experiment config at root level
- **Selective replacement**: `override` keyword replaces specific default components
- **Parameter hub**: `params` section provides values resolved by interpolations
- **Complex objects**: Use `_target_` with initialization parameters for non-primitive objects

## Step 6: Custom Transform Implementation

Implement custom preprocessing, augmentation, or model logic as callable classes.

**Example**: Masking transforms for language modeling (`ml_core/transforms/diffusion/mdlm.py`)

```python
import torch

class ToBool:
    """Convert tensor to boolean dtype."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.bool()


class SampleMask:
    """Sample random mask for masked language modeling."""

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        # Sample k ~ U[1, L]
        k = torch.randint(1, seq_length + 1, (batch_size,), device=device)

        # Sample mask ~ B(k/L)
        p = k.float() / seq_length
        probs = p.unsqueeze(1).expand(batch_size, seq_length)
        mask = torch.bernoulli(probs).bool()

        return mask


class MaskInputIds:
    """Mask input_ids at specified positions."""

    def __init__(self, mask_token_id: int = None, tokenizer = None):
        if mask_token_id is not None:
            self.mask_token_id = mask_token_id
        elif tokenizer is not None:
            self.mask_token_id = tokenizer.mask_token_id
        else:
            raise ValueError("Provide mask_token_id or tokenizer")

    def __call__(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask] = self.mask_token_id
        return masked_input_ids
```

**Key Concepts**:

- **Callable interface**: Implement `__call__` method for function-like behavior
- **Tensor operations**: Accept and return PyTorch tensors for GPU compatibility
- **Stateful transforms**: Inherit from `torch.nn.Module` if the transform has learnable parameters

## Pipeline Execution Flow

This example shows the complete data flow for the MDLM task:

```
1. DataModule loads data
   └─> Dataset loader (HuggingFace/PyTorch)

2. Transform pipeline preprocesses batch
   └─> Tokenizer: raw_text -> input_ids, attention_mask
   └─> Type conversion: attention_mask (long -> bool)

3. Forward function processes batch
   └─> Data augmentation: input_ids -> mask -> masked_input_ids
   └─> Model inference: (masked_input_ids, attention_mask) -> logits
   └─> Post-processing: reshape logits for loss computation

4. Criterion computes loss
   └─> Loss function: (predictions, targets) -> scalar loss

5. Optimizer updates parameters
   └─> Gradient-based optimization with learning rate scheduling
```

The same pattern applies to other tasks (e.g., image classification, object detection) by replacing task-specific components.

## Best Practices

### 1. Use Params for Shared Values

**Important**: Use consistent naming conventions (`params.model.*`, `params.data.*`) to enable automatic test optimization. Tests can override these parameters globally for faster execution.

```yaml
params:
  model:
    dim: 16
    depth: 2
    heads: 4
    num_tokens: 30522  # Always name vocab size as 'num_tokens'
  data:
    max_length: 128
    tokenizer: ???      # Always nest tokenizer under 'data'

# Reference in multiple places using interpolations
model:
  dim: ${params.model.dim}
  num_tokens: ${params.model.num_tokens}
data:
  max_length: ${params.data.max_length}
  tokenizer: ${params.data.tokenizer}
```

**Why this matters**: Test configurations can override `params.*` globally to use smaller models and faster tokenizers, making your entire test suite run 100× faster without modifying individual configs.

### 2. Control Defaults Order with `_self_`

```yaml
defaults:
  - _self_  # Current config values take priority
  - imported_config
```

### 3. Use `@` to Place Imports

```yaml
defaults:
  # Import encoder.yaml into forward_fn.network.transform
  - /model/transformer@forward_fn.network.transform: encoder
```

### 4. Keep Transforms Modular

- One transform = one responsibility
- Use `ComposeTransform` to chain
- Use `WrapTransform` to handle key mapping

### 5. Type Conversions in Transforms

```yaml
convert_mask:
  _target_: ml_core.transforms.base.WrapTransform
  transform:
    _target_: ml_core.transforms.diffusion.mdlm.ToBool
  mapping:
    attention_mask: tensor
  new_key: attention_mask
```

## Running the Pipeline

```bash
# Train with experiment config
python ml_core/train.py experiment=<experiment_name>

# Override hyperparameters from command line
python ml_core/train.py experiment=<experiment_name> \
  params.model.dim=32 \
  trainer.max_epochs=10 \
  data.batch_size=64

# Evaluate a trained checkpoint
python ml_core/eval.py experiment=<experiment_name> \
  ckpt_path=logs/train/runs/<run_id>/checkpoints/last.ckpt
```

## Summary

This guide demonstrates a modular pipeline architecture:

1. **Data Configuration**: Use `BaseDataModule` with any dataset source (HuggingFace, custom, etc.)
2. **Transform Composition**: Chain preprocessing steps with `ComposeTransform` and route keys with `WrapTransform`
3. **Model Components**: Define reusable architecture blocks with parameter interpolations
4. **Forward Pipeline**: Compose data flow using `ComposeTransform` and import sub-configs via defaults
5. **Experiment Binding**: Centralize hyperparameters in `params` and override components as needed
6. **Custom Logic**: Implement domain-specific transforms as simple callable classes

**Benefits**:

- **Modularity**: Swap components without touching other parts
- **Reusability**: Same transforms/models work across experiments
- **Testability**: Test each component independently
- **Composability**: Build complex pipelines from simple parts
- **Configurability**: Change behavior via YAML without code changes

This pattern scales from simple classification tasks to complex multi-stage pipelines.
