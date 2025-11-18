import pytest
import torch

from ml_core.transforms.base import ComposeTransform, RenameTransform, WrapTransform


def test_rename_transform():
    """Test RenameTransform renames batch keys correctly."""
    batch = {"old_key_1": 1, "old_key_2": 2, "keep_key": 3}
    transform = RenameTransform(mapping={"new_key_1": "old_key_1", "new_key_2": "old_key_2"})
    result = transform(batch)

    assert "new_key_1" in result
    assert "new_key_2" in result
    assert result["new_key_1"] == 1
    assert result["new_key_2"] == 2
    assert "keep_key" not in result  # Only mapped keys are returned


def test_compose_transform():
    """Test ComposeTransform applies multiple transforms sequentially."""
    batch = {"x": 1, "y": 2}

    def add_z(x, y):
        return x + y

    def add_w(z):
        return z * 2

    transform = ComposeTransform(
        add_z=WrapTransform(
            transform=add_z,
            new_key="z",
            mapping={"x": "x", "y": "y"},
        ),
        add_w=WrapTransform(
            transform=add_w,
            new_key="w",
            mapping={"z": "z"},
        ),
    )

    result = transform(batch)
    assert result["x"] == 1
    assert result["y"] == 2
    assert result["z"] == 3
    assert result["w"] == 6


def test_wrap_transform_basic():
    """Test WrapTransform with a basic callable."""
    batch = {"a": 10, "b": 20}

    def add(a, b):
        return a + b

    transform = WrapTransform(
        transform=add,
        new_key="sum",
        mapping={"a": "a", "b": "b"},
    )

    result = transform(batch)
    assert result["a"] == 10
    assert result["b"] == 20
    assert result["sum"] == 30


def test_wrap_transform_with_transform_kwargs():
    """Test WrapTransform with additional kwargs."""
    batch = {"x": 5}

    def multiply(x, factor):
        return x * factor

    transform = WrapTransform(
        transform=multiply,
        new_key="result",
        mapping={"x": "x"},
        transform_kwargs={"factor": 3},
    )

    result = transform(batch)
    assert result["x"] == 5
    assert result["result"] == 15


def test_wrap_transform_without_new_key():
    """Test WrapTransform that returns a dict directly."""
    batch = {"value": 100}

    def create_dict(value):
        return {"doubled": value * 2, "tripled": value * 3}

    transform = WrapTransform(
        transform=create_dict,
        new_key=None,
        mapping={"value": "value"},
    )

    result = transform(batch)
    assert result["value"] == 100
    assert result["doubled"] == 200
    assert result["tripled"] == 300


def test_wrap_transform_with_method_name():
    """Test WrapTransform with method_name parameter to extract a method from a class."""
    batch = {"input": torch.tensor([1.0, 2.0, 3.0])}

    # Create a simple class with a method
    class SimpleProcessor:
        def process(self, input):
            return input * 2

        def other_method(self, input):
            return input + 10

    processor = SimpleProcessor()

    # Test extracting the 'process' method
    transform = WrapTransform(
        transform=processor,
        new_key="output",
        mapping={"input": "input"},
        method_name="process",
    )

    result = transform(batch)
    assert torch.allclose(result["output"], torch.tensor([2.0, 4.0, 6.0]))

    # Test extracting a different method
    transform2 = WrapTransform(
        transform=processor,
        new_key="output",
        mapping={"input": "input"},
        method_name="other_method",
    )

    result2 = transform2(batch)
    assert torch.allclose(result2["output"], torch.tensor([11.0, 12.0, 13.0]))


def test_wrap_transform_method_name_with_kwargs():
    """Test WrapTransform with method_name and transform_kwargs."""
    batch = {"data": 5}

    class Calculator:
        def compute(self, data, multiplier, offset):
            return data * multiplier + offset

    calc = Calculator()

    transform = WrapTransform(
        transform=calc,
        new_key="result",
        mapping={"data": "data"},
        method_name="compute",
        transform_kwargs={"multiplier": 10, "offset": 3},
    )

    result = transform(batch)
    assert result["data"] == 5
    assert result["result"] == 53  # 5 * 10 + 3


def test_wrap_transform_without_method_name():
    """Test WrapTransform without method_name behaves as before."""
    batch = {"x": 7}

    def square(x):
        return x**2

    transform = WrapTransform(
        transform=square,
        new_key="squared",
        mapping={"x": "x"},
        method_name=None,
    )

    result = transform(batch)
    assert result["x"] == 7
    assert result["squared"] == 49
