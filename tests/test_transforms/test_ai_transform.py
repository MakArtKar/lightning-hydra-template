"""Unit tests for AITransformWrapper."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ml_core.transforms.ai_transform import AITransformWrapper


@contextmanager
def mock_ai_transform():
    """Context manager to set up common mocks for AITransformWrapper tests."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch.object(AITransformWrapper, '_generate_code') as mock_generate:
            with patch.object(AITransformWrapper, '_import_transform') as mock_import:
                with tempfile.TemporaryDirectory() as temp_dir:
                    mock_transform = Mock(return_value=42)
                    mock_import.return_value = mock_transform
                    yield mock_generate, mock_import, temp_dir


class TestAITransformWrapper:
    """Test cases for AITransformWrapper."""

    def test_init_with_simple_prompt(self):
        """Test initialization with a simple prompt."""
        with mock_ai_transform() as (mock_generate, mock_import, temp_dir):
            # Mock file existence to force code generation
            with patch('pathlib.Path.exists', return_value=False):
                wrapper = AITransformWrapper(
                    transform_prompt="def add_one(x: float) -> float:\n    \"\"\"Add 1 to input.\"\"\"\n    return x + 1",
                    new_key="result",
                    path=temp_dir
                )
            
            assert wrapper.new_key == "result"
            assert wrapper.mapping is None
            mock_generate.assert_called_once()
            mock_import.assert_called_once()

    def test_init_with_mapping(self):
        """Test initialization with mapping."""
        with mock_ai_transform() as (mock_generate, mock_import, temp_dir):
            mapping = {"input_value": "x"}
            wrapper = AITransformWrapper(
                transform_prompt="def multiply_by_two(x: float) -> float:\n    \"\"\"Multiply by 2.\"\"\"\n    return x * 2",
                new_key="doubled",
                mapping=mapping,
                path=temp_dir
            )
            
            assert wrapper.mapping == mapping
            assert wrapper.new_key == "doubled"

    def test_init_with_custom_path(self):
        """Test initialization with custom path."""
        with mock_ai_transform() as (mock_generate, mock_import, temp_dir):
            wrapper = AITransformWrapper(
                transform_prompt="def square_input(x: float) -> float:\n    \"\"\"Square the input.\"\"\"\n    return x * x",
                new_key="squared",
                path=temp_dir
            )
            
            # Check that the path was used in the call
            call_args = mock_import.call_args[0][0]
            assert temp_dir in str(call_args)


    def test_call_without_mapping(self):
        """Test __call__ method without mapping."""
        with mock_ai_transform() as (mock_generate, mock_import, temp_dir):
            # Mock transform that adds 1 to input
            mock_transform = Mock(side_effect=lambda **kwargs: kwargs['input'] + 1)
            mock_import.return_value = mock_transform
            
            wrapper = AITransformWrapper(
                transform_prompt="def add_one(x: float) -> float:\n    \"\"\"Add 1.\"\"\"\n    return x + 1",
                new_key="result",
                path=temp_dir
            )
            
            batch = {"input": 5, "other": "data"}
            result = wrapper(batch)
            
            expected = {"input": 5, "other": "data", "result": 6}
            assert result == expected
            mock_transform.assert_called_once_with(input=5, other="data")

    def test_call_with_mapping(self):
        """Test __call__ method with mapping."""
        with mock_ai_transform() as (mock_generate, mock_import, temp_dir):
            # Mock transform that multiplies by 2
            mock_transform = Mock(side_effect=lambda **kwargs: kwargs['x'] * 2)
            mock_import.return_value = mock_transform
            
            mapping = {"value": "x"}
            wrapper = AITransformWrapper(
                transform_prompt="def multiply_by_two(x: float) -> float:\n    \"\"\"Multiply by 2.\"\"\"\n    return x * 2",
                new_key="doubled",
                mapping=mapping,
                path=temp_dir
            )
            
            batch = {"value": 7, "other": "data"}
            result = wrapper(batch)
            
            expected = {"value": 7, "other": "data", "doubled": 14}
            assert result == expected
            mock_transform.assert_called_once_with(x=7)

    def test_missing_openai_key(self):
        """Test error when OpenAI API key is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is required"):
                    AITransformWrapper(
                        transform_prompt="def test_function(x: float) -> float:\n    \"\"\"Test function.\"\"\"\n    return x",
                        new_key="result",
                        path=temp_dir
                    )

    def test_openai_import_error(self):
        """Test error when openai package is not installed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                with patch('builtins.__import__', side_effect=ImportError("No module named 'openai'")):
                    with pytest.raises(ImportError, match="openai package is required"):
                        AITransformWrapper(
                            transform_prompt="def test_function(x: float) -> float:\n    \"\"\"Test function.\"\"\"\n    return x",
                            new_key="result",
                            path=temp_dir
                        )

    def test_generate_code_success(self):
        """Test successful code generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                # Mock OpenAI response
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = """
def add_one(x: float) -> float:
    \"\"\"Add 1 to the input.\"\"\"
    return x + 1
"""
                
                # Mock the OpenAI import and client
                mock_openai_module = Mock()
                mock_client = Mock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai_module.OpenAI.return_value = mock_client
                
                with patch.dict('sys.modules', {'openai': mock_openai_module}):
                    with patch.object(AITransformWrapper, '_import_transform') as mock_import:
                        mock_transform = Mock(return_value=1)
                        mock_import.return_value = mock_transform
                        
                        # Mock file existence to force code generation
                        with patch('pathlib.Path.exists', return_value=False):
                            wrapper = AITransformWrapper(
                                transform_prompt="def add_one(x: float) -> float:\n    \"\"\"Add 1 to input.\"\"\"\n    return x + 1",
                                new_key="result",
                                path=temp_dir
                            )
                        
                        # Check that OpenAI was called
                        mock_client.chat.completions.create.assert_called_once()

    def test_import_transform_success(self):
        """Test successful transform import."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                with patch.object(AITransformWrapper, '_generate_code'):
                    with patch('builtins.__import__') as mock_import:
                        mock_module = Mock()
                        mock_module.square_input = Mock(return_value=25)
                        mock_import.return_value = mock_module
                        
                        wrapper = AITransformWrapper(
                            transform_prompt="def square_input(x: float) -> float:\n    \"\"\"Square input.\"\"\"\n    return x * x",
                            new_key="squared",
                            path=temp_dir
                        )
                        
                        # Test the transform
                        batch = {"input": 5}
                        result = wrapper(batch)
                        
                        assert result["squared"] == 25

    def test_import_transform_missing_main(self):
        """Test error when imported module doesn't have main function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                with patch.object(AITransformWrapper, '_generate_code'):
                    with patch('builtins.__import__') as mock_import:
                        mock_module = Mock()
                        # Remove the function attribute
                        del mock_module.test_function
                        mock_import.return_value = mock_module
                        
                        with pytest.raises(ImportError, match="Failed to import generated transform"):
                            AITransformWrapper(
                                transform_prompt="def test_function(x: float) -> float:\n    \"\"\"Test function.\"\"\"\n    return x",
                                new_key="result",
                                path=temp_dir
                            )

    def test_function_name_extraction(self):
        """Test that function names are extracted correctly from prompts."""
        with mock_ai_transform() as (mock_generate, mock_import, temp_dir):
            # Mock file existence to force code generation
            with patch('pathlib.Path.exists', return_value=False):
                wrapper1 = AITransformWrapper(
                    transform_prompt="def add_one(x: float) -> float:\n    \"\"\"Add 1 to input.\"\"\"\n    return x + 1",
                    new_key="result1",
                    path=temp_dir
                )
                wrapper2 = AITransformWrapper(
                    transform_prompt="def multiply_by_two(x: float) -> float:\n    \"\"\"Multiply by 2.\"\"\"\n    return x * 2",
                    new_key="result2",
                    path=temp_dir
                )
            
            # Check that _generate_code was called with correct function names
            assert mock_generate.call_count == 2
            
            # First call should have "add_one" as function name
            call1_args = mock_generate.call_args_list[0][0]
            assert call1_args[1] == "add_one"  # function_name is 2nd argument
            
            # Second call should have "multiply_by_two" as function name
            call2_args = mock_generate.call_args_list[1][0]
            assert call2_args[1] == "multiply_by_two"  # function_name is 2nd argument

    def test_invalid_prompt_structure(self):
        """Test error when prompt doesn't start with function definition."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="Transform prompt must start with a function definition"):
                AITransformWrapper(
                    transform_prompt="This is not a function definition",
                    new_key="result"
                )

    def test_model_parameter(self):
        """Test that model parameter is passed correctly."""
        with mock_ai_transform() as (mock_generate, mock_import, temp_dir):
            # Mock file existence to force code generation
            with patch('pathlib.Path.exists', return_value=False):
                wrapper = AITransformWrapper(
                    transform_prompt="def test_func(x: float) -> float:\n    \"\"\"Test function.\"\"\"\n    return x",
                    new_key="result",
                    model="gpt-4",
                    path=temp_dir
                )
            
            # Check that _generate_code was called with the correct model
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args[0]
            assert call_args[3] == "gpt-4"  # model is the 4th argument

    def test_default_model(self):
        """Test that default model is used when not specified."""
        with mock_ai_transform() as (mock_generate, mock_import, temp_dir):
            # Mock file existence to force code generation
            with patch('pathlib.Path.exists', return_value=False):
                wrapper = AITransformWrapper(
                    transform_prompt="def test_func(x: float) -> float:\n    \"\"\"Test function.\"\"\"\n    return x",
                    new_key="result",
                    path=temp_dir
                )
            
            # Check that _generate_code was called with the default model
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args[0]
            assert call_args[3] == "gpt-5-nano"  # model is the 4th argument
