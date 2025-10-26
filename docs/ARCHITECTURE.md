# Architecture

This document describes the architecture of the Lightning-Hydra ML training framework.

## Overview

This is a flexible framework for training and evaluating ML models as a Python package or template, powered by [PyTorch Lightning](https://lightning.ai/) and [Hydra](https://hydra.cc/). The framework implements core Lightning components (`LightningDataModule` and `LightningModule`) with a composable design that enables rapid experimentation through configuration.

## Core Design Principles

The framework follows these key design principles:

1. **Composition over Inheritance**: Components are composed using configuration rather than extending base classes
2. **Configuration-Driven**: All experiments are defined via Hydra YAML configs, enabling reproducibility and version control
3. **Flexibility**: Easy to swap data transforms, model architectures, losses, metrics, and hyperparameters without changing code
4. **Batch-Based Operations**: All components operate on dictionary-based batches for maximum flexibility

## Project Structure

```
ml_core/
├── train.py                    # Training entrypoint
├── eval.py                     # Evaluation entrypoint
├── data/
│   └── base_datamodule.py     # Generic LightningDataModule
├── models/
│   ├── base_module.py         # Generic LightningModule
│   ├── utils.py               # Loss and metric compositions
│   └── components/            # Model architectures
├── transforms/
│   ├── base.py                # Generic dict-based transforms
│   └── image.py               # Image-specific transforms
└── utils/                      # Helper utilities

configs/
├── train.yaml                  # Main training config
├── eval.yaml                   # Main evaluation config
├── data/                       # Dataset configurations
├── model/                      # Model configurations
├── trainer/                    # PyTorch Lightning trainer configs
├── callbacks/                  # Callback configurations
├── logger/                     # Logger configurations
├── debug/                      # Debug presets
└── experiment/                 # Experiment-specific configs
```

## Core Components

### 1. Entry Points

#### `ml_core/train.py`

The main training entrypoint that:
- Loads Hydra configuration from `configs/train.yaml`
- Instantiates all components (data, model, trainer, callbacks, loggers)
- Runs training with `trainer.fit()`
- Optionally runs testing with `trainer.test()`
- Returns metrics for hyperparameter optimization

**Usage:**
```bash
python ml_core/train.py --config-dir configs
```

#### `ml_core/eval.py`

The evaluation entrypoint that:
- Loads Hydra configuration from `configs/eval.yaml`
- Requires a checkpoint path (`ckpt_path`)
- Instantiates components and runs `trainer.test()`
- Useful for evaluating saved models on test sets

**Usage:**
```bash
python ml_core/eval.py --config-dir configs ckpt_path=/path/to/checkpoint.ckpt
```

### 2. Data Module (`BaseDataModule`)

Location: `ml_core/data/base_datamodule.py`

A generic `LightningDataModule` that:
- Wraps Hugging Face `DatasetDict` for data loading
- Automatically splits train data into train/validation using `val_ratio`
- Applies optional transforms on-the-fly via `with_transform`
- Handles batch size distribution across devices in distributed training
- Creates PyTorch `DataLoader` instances for train/val/test

**Key Parameters:**
- `hf_dict_dataset`: Hugging Face DatasetDict (loaded via config)
- `val_ratio`: Fraction of training data used for validation (default: 0.1)
- `transform`: Optional callable transform applied to batches
- `**dataloader_kwargs`: Arguments passed to DataLoader (batch_size, num_workers, etc.)

### 3. Lightning Module (`BaseLitModule`)

Location: `ml_core/models/base_module.py`

A generic `LightningModule` that wires together:
- **Forward function**: Callable that processes batches
- **Losses**: `CriterionsComposition` for computing multiple weighted losses
- **Metrics**: Optional `MetricsComposition` for tracking performance
- **Optimizer**: Factory function returning configured optimizer
- **Scheduler**: Optional LR scheduler factory

**Key Parameters:**
- `forward_fn`: Callable mapping batch dict to output dict
- `criterions`: Composition of loss functions with weights
- `optimizer`: Partial optimizer constructor
- `scheduler`: Optional partial scheduler constructor
- `metrics`: Optional composition of metrics
- `tracked_metric_name`: Metric to track for best model selection
- `compile`: Whether to use `torch.compile()` for speedup

**Workflow:**
1. `forward()` calls `forward_fn(batch)` to get predictions
2. `model_step()` computes losses and metrics for train/val/test
3. Logs all losses and metrics with `sync_dist=True` for DDP
4. Tracks best validation metric for model checkpointing

### 4. Transforms

Location: `ml_core/transforms/base.py`

Dictionary-based transforms that enable flexible data preprocessing and augmentation:

#### `ComposeTransform`
Composes multiple transforms sequentially, merging outputs into the batch dict.

```python
transform = ComposeTransform(
    step1=SomeTransform(),
    step2=AnotherTransform()
)
```

#### `WrapTransform`
Wraps any callable, mapping batch keys to function arguments and storing output under a new key.

```python
transform = WrapTransform(
    transform=some_function,
    new_key="output",
    mapping={"batch_key": "function_arg"}
)
```

#### `RenameTransform`
Creates a view of the batch with renamed keys.

```python
transform = RenameTransform({"old_key": "new_key"})
```

### 5. Loss and Metric Compositions

Location: `ml_core/models/utils.py`

#### `CriterionsComposition`
Composes multiple loss functions with:
- Per-loss input mapping from batch keys to criterion arguments
- Weighted sum producing a `"total"` loss
- Individual loss tracking

```yaml
criterions:
  _target_: ml_core.models.utils.CriterionsComposition
  criterions:
    ce:
      _target_: torch.nn.CrossEntropyLoss
    mse:
      _target_: torch.nn.MSELoss
  weights:
    ce: 1.0
    mse: 0.5
  mapping:
    ce:
      output: input
      label: target
```

#### `MetricsComposition`
Extends `torchmetrics.MetricCollection` with per-metric input mapping from batch keys.

```yaml
metrics:
  _target_: ml_core.models.utils.MetricsComposition
  metrics:
    accuracy:
      _target_: torchmetrics.Accuracy
      task: multiclass
      num_classes: 10
  mapping:
    accuracy:
      prediction: preds
      label: target
```

## Configuration System

### Hydra Configuration

The framework uses Hydra for hierarchical configuration composition. Main config file: `configs/train.yaml`

#### Config Groups

- **data**: Dataset and dataloader configurations
- **model**: Model architecture, losses, metrics, optimizer configurations
- **trainer**: PyTorch Lightning Trainer settings (devices, precision, etc.)
- **callbacks**: Lightning callbacks (checkpointing, early stopping, etc.)
- **logger**: Experiment loggers (wandb, tensorboard, mlflow, etc.)
- **paths**: Directory paths for data, logs, and outputs
- **experiment**: Pre-configured experiment setups
- **debug**: Debug presets (see below)

#### Config Overrides

Override any configuration from command line:

```bash
# Change model and data
python ml_core/train.py data=mnist model=mnist

# Override specific parameters
python ml_core/train.py trainer.max_epochs=10 data.batch_size=64

# Use experiment config
python ml_core/train.py experiment=example

# Use debug preset
python ml_core/train.py debug=fdr
```

### Debug Presets

Location: `configs/debug/`

Debug configurations help during development and troubleshooting:

#### `debug=default`
- **Purpose**: Standard debugging setup for 1 full epoch
- **Settings**:
  - Disables callbacks and loggers
  - Runs 1 epoch on CPU with 1 device
  - Sets logging level to DEBUG
  - Enables anomaly detection (raises exception on NaN/inf)
  - Disables multiprocessing (`num_workers=0`)
  - Stores logs in separate `debug/` folder

```bash
python ml_core/train.py debug=default
```

#### `debug=fdr` (Fast Dev Run)
- **Purpose**: Quick sanity check - runs 1 train, 1 validation, and 1 test step
- **Settings**: Inherits from `default` + sets `fast_dev_run=true`
- **Use Case**: Verify code runs without errors before full training

```bash
python ml_core/train.py debug=fdr
```

#### `debug=limit`
- **Purpose**: Train on small subset of data
- **Settings**:
  - Uses 1% of training data
  - Uses 5% of validation/test data
  - Runs for 3 epochs
- **Use Case**: Quick iterations when testing changes

```bash
python ml_core/train.py debug=limit
```

#### `debug=overfit`
- **Purpose**: Overfit to small number of batches to verify model can learn
- **Settings**:
  - Overfits to 3 batches for 20 epochs
  - Disables callbacks (checkpointing, early stopping)
- **Use Case**: Debugging model implementation - if it can't overfit, there's a bug

```bash
python ml_core/train.py debug=overfit
```

#### `debug=profiler`
- **Purpose**: Profile execution time of training components
- **Settings**:
  - Runs 1 epoch with profiler enabled
  - Options: `"simple"`, `"advanced"`, or `"pytorch"` profiler
- **Use Case**: Identify performance bottlenecks

```bash
python ml_core/train.py debug=profiler
```

## Example Workflow: MNIST Classification

### 1. Data Configuration (`configs/data/mnist.yaml`)

```yaml
_target_: ml_core.data.base_datamodule.BaseDataModule
hf_dict_dataset:
  _target_: datasets.load_dataset
  _args_:
    - ylecun/mnist
val_ratio: 0.1

transform:
  _target_: ml_core.transforms.base.ComposeTransform
  to_tensor:
    _target_: ml_core.transforms.image.TorchVisionTransform
    transform:
      _target_: torchvision.transforms.ToTensor
    key: image
  normalize:
    _target_: ml_core.transforms.image.TorchVisionTransform
    transform:
      _target_: torchvision.transforms.Normalize
      mean: [0.1307]
      std: [0.3081]
    key: image

batch_size: 128
num_workers: 8
```

### 2. Model Configuration (`configs/model/mnist.yaml`)

```yaml
_target_: ml_core.models.base_module.BaseLitModule

forward_fn:
  _target_: ml_core.transforms.base.ComposeTransform
  network:
    _target_: ml_core.transforms.base.WrapTransform
    transform:
      _target_: ml_core.models.components.simple_dense_net.SimpleDenseNet
      input_size: 784
      output_size: 10
    mapping:
      image: x
    new_key: output
  argmax:
    _target_: ml_core.transforms.base.WrapTransform
    transform:
      _target_: torch.argmax
      _partial_: true
      dim: -1
    mapping:
      output: input
    new_key: prediction

criterions:
  _target_: ml_core.models.utils.CriterionsComposition
  criterions:
    ce:
      _target_: torch.nn.CrossEntropyLoss
  weights:
    ce: 1.0
  mapping:
    ce:
      output: input
      label: target

metrics:
  _target_: ml_core.models.utils.MetricsComposition
  metrics:
    accuracy:
      _target_: torchmetrics.Accuracy
      task: multiclass
      num_classes: 10
  mapping:
    accuracy:
      prediction: preds
      label: target

tracked_metric_name: accuracy

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
```

### 3. Run Training

```bash
python ml_core/train.py --config-dir configs
```

## Conducting New Experiments

To run a new experiment:

1. **Choose or create a dataset configuration** in `configs/data/`
   - Specify Hugging Face dataset
   - Define data transforms (normalization, augmentation, etc.)
   - Set dataloader parameters

2. **Choose or create a model configuration** in `configs/model/`
   - Define `forward_fn` (network architecture + post-processing)
   - Specify losses in `criterions` with weights and mappings
   - Define metrics to track
   - Configure optimizer and scheduler

3. **Optionally create an experiment config** in `configs/experiment/`
   - Combines specific data, model, trainer settings
   - Version control best hyperparameters

4. **Run training**
   ```bash
   python ml_core/train.py data=<your_data> model=<your_model>
   ```

5. **Evaluate on test set**
   ```bash
   python ml_core/eval.py ckpt_path=<checkpoint_path>
   ```

## Advanced Features

### Hyperparameter Optimization

Use Hydra's multirun with Optuna:

```bash
python ml_core/train.py -m hparams_search=optuna \
    model.optimizer.lr=tag(log, interval(0.0001, 0.1))
```

### Multi-GPU Training

Configure trainer for distributed training:

```bash
python ml_core/train.py trainer=ddp trainer.devices=4
```

### Logging

Choose from multiple experiment trackers:

```bash
python ml_core/train.py logger=wandb
python ml_core/train.py logger=tensorboard
python ml_core/train.py logger=mlflow
```

### Custom Components

Add custom components by:
1. Creating the component class/function
2. Adding it to the appropriate module
3. Referencing it via `_target_` in configs

## Batch Dictionary Convention

All components operate on dictionary batches with the following conventions:

- Data loaders produce dicts with raw data (e.g., `{"image": ..., "label": ...}`)
- Transforms augment the dict (e.g., add `"output"`, `"prediction"`)
- Losses and metrics pull specific keys via `mapping` configurations
- This design enables flexible composition without rigid interfaces

## Distributed Training Support

The framework handles distributed training automatically:
- Batch sizes are divided across devices
- Metrics are synchronized with `sync_dist=True`
- Supports DDP, FSDP, and other Lightning strategies
- Configure via `trainer` configs

## Summary

This architecture provides:
- ✅ **Flexibility**: Swap components via configuration
- ✅ **Reproducibility**: All experiments defined in version-controlled configs
- ✅ **Scalability**: Built-in distributed training support
- ✅ **Debuggability**: Multiple debug presets for development
- ✅ **Extensibility**: Easy to add new models, losses, metrics, transforms
- ✅ **Best Practices**: Leverages Lightning's training loop and Hydra's config management

