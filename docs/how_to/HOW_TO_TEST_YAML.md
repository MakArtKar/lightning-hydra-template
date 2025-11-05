# How to Test YAML Configs

This guide describes best practices for creating unit tests for YAML configuration files using Hydra.

## Rules

### 1. Use Hydra Instantiate - Not `__init__`

All instances should be generated with `hydra.utils.instantiate()` of the tested YAML config. **Never use `__init__` methods of classes directly.**

### 2. Test Only What You Need

Use only the configs you want to test. Don't override with other configs unless necessary.

### 3. Use Absolute Paths with PROJECT_ROOT

Always use `PROJECT_ROOT` to obtain the absolute path to the configs root:

```python
import os
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
```

### 4. Config Path Structure

Use this approach to avoid conflicts with `_target_`:

```python
config_path = os.path.join(PROJECT_ROOT, "configs")
config_name = "data/transform/tokenizer"  # path to tested config
```

With this approach, you can add extra parameters to the root (like `params` key) without conflicting with `_target_`. Extract the desired object using the same path and pass it to instantiate.

### 5. Fixtures vs Functions

- **Create a fixture** that returns the desired object for reusable test setup
- **Use a function** if you need different override params across tests

### 6. Handle Non-Primitive Objects

For non-primitive objects (like tokenizers), use Hydra overrides with `_target_`:

```python
overrides = [
    "++params.data.tokenizer._target_=transformers.AutoTokenizer.from_pretrained",
    "++params.data.tokenizer.pretrained_model_name_or_path=bert-base-uncased",
]
```

## Example

From `tests/test_data/test_transform/test_tokenizer.py`:

```python
import os
import pytest
import rootutils
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Setup PROJECT_ROOT
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
PROJECT_ROOT = os.environ["PROJECT_ROOT"]


@pytest.fixture(scope="module")
def tokenizer_transform():
    """Fixture that loads config and returns a WrapTransform object."""
    GlobalHydra.instance().clear()

    config_path = os.path.join(PROJECT_ROOT, "configs")

    # Define test parameters
    max_length = 128
    tokenizer_model = "bert-base-uncased"

    with initialize_config_dir(version_base="1.3", config_dir=config_path):
        # Load the config with overrides
        cfg = compose(
            config_name="data/transform/tokenizer",
            overrides=[
                f"++params.data.max_length={max_length}",
                f"++params.data.tokenizer._target_=transformers.AutoTokenizer.from_pretrained",
                f"++params.data.tokenizer.pretrained_model_name_or_path={tokenizer_model}",
            ],
        )

        # Instantiate using Hydra
        from hydra.utils import instantiate
        transform = instantiate(cfg.data.transform)

        yield transform

    GlobalHydra.instance().clear()


def test_basic(tokenizer_transform):
    """Test basic functionality."""
    # Use the fixture
    result = tokenizer_transform({"text": "test"})
    assert "input_ids" in result
```

## Key Patterns

1. **Clear GlobalHydra** before and after tests to avoid conflicts
2. **Use `initialize_config_dir`** with absolute `config_path`
3. **Use `compose()`** to load configs with overrides
4. **Extract the target** from config structure (e.g., `cfg.data.transform`)
5. **Use `instantiate()`** to create the object from config
6. **Use `++` prefix** to force-add new keys in struct mode

## Common Pitfalls

- ❌ Creating objects with `__init__` directly
- ❌ Using relative paths for configs
- ❌ Trying to store non-primitive objects in OmegaConf
- ❌ Not clearing GlobalHydra between tests
- ✅ Always use `hydra.utils.instantiate()`
- ✅ Use `_target_` overrides for complex objects
- ✅ Use absolute paths with `PROJECT_ROOT`
