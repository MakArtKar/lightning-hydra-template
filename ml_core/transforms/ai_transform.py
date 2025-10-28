"""AI-powered transform wrapper that generates code from prompts."""

import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import rootutils

from .base import WrapTransform

# Setup project root like in train.py
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

# Template prompt for code generation
TEMPLATE_PROMPT = """Generate a Python function that implements the following functionality:

{transform_prompt}

The function should:
1. Be named '{function_name}'
2. Take keyword arguments that will be mapped from batch keys
3. Return a single value (not a dict)
4. Be a pure function (no side effects)
5. Include proper type hints
6. Include a docstring

Return only the Python code, no explanations or markdown formatting."""


class AITransformWrapper(WrapTransform):
    """AI-powered transform wrapper that generates code from text prompts.

    This class takes a text description of a function, generates Python code using an LLM, saves it
    to a file, and wraps it as a transform.

    :param transform_prompt: Text description of the function to generate. Should start with "def
        my_function_name".
    :param new_key: Key to place the function output under in the batch.
    :param path: Optional subdirectory path for generated files (default: "ai_generations/").
    :param mapping: Optional mapping from batch keys to function argument names.
    :param force: Whether to regenerate the code even if file exists.
    :param api_kwargs: Optional dictionary of parameters to pass to OpenAI API. Defaults to
        model="gpt-3.5-turbo", temperature=0.1, max_tokens=1000.
    """

    def __init__(
        self,
        transform_prompt: str,
        new_key: str,
        path: str | None = None,
        mapping: Mapping[str, str] | None = None,
        force: bool = False,
        api_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the AI transform wrapper."""
        # Extract function name from prompt
        function_name = self._extract_function_name(transform_prompt)

        # Generate filename from function name
        filename = f"{function_name}.py"

        # Set up path
        if path is None:
            # Default to ai_generations/ subdirectory
            full_path = PROJECT_ROOT / "ml_core" / "transforms" / "ai_generations"
        else:
            # Use provided path (can be absolute or relative)
            full_path = (
                Path(path)
                if Path(path).is_absolute()
                else PROJECT_ROOT / "ml_core" / "transforms" / path
            )
        full_path.mkdir(parents=True, exist_ok=True)

        file_path = full_path / filename

        # Generate code if file doesn't exist or force is True
        if not file_path.exists() or force:
            self._generate_code(transform_prompt, function_name, file_path, api_kwargs)

        # Import the generated module
        transform = self._import_transform(file_path, function_name)

        # Initialize parent WrapTransform
        super().__init__(transform, new_key, mapping)

    def _extract_function_name(self, transform_prompt: str) -> str:
        """Extract function name from the transform prompt.

        Expected format: def my_function_name(param1: type1, param2: type2, ...) -> type_out:

        :param transform_prompt: The prompt containing the function definition
        :return: The extracted function name
        :raises ValueError: If the prompt doesn't start with a valid function definition
        """
        # Look for function definition pattern
        pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        match = re.search(pattern, transform_prompt.strip())

        if not match:
            raise ValueError(
                "Transform prompt must start with a function definition like 'def my_function_name(param1: type1, ...) -> type_out:'. "
                f"Got: {transform_prompt[:100]}{'...' if len(transform_prompt) > 100 else ''}"
            )

        return match.group(1)

    def _generate_code(
        self,
        transform_prompt: str,
        function_name: str,
        file_path: Path,
        api_kwargs: dict[str, Any] | None,
    ) -> None:
        """Generate Python code using OpenAI API and save to file."""
        # Get OpenAI API key from environment first
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for AITransformWrapper. Install with: pip install openai"
            )

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Set default API kwargs
        default_kwargs = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_tokens": 1000,
        }

        # Merge with provided kwargs (provided kwargs take precedence)
        final_kwargs = default_kwargs.copy()
        if api_kwargs:
            final_kwargs.update(api_kwargs)

        # Generate code using OpenAI
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": TEMPLATE_PROMPT.format(
                            transform_prompt=transform_prompt, function_name=function_name
                        ),
                    }
                ],
                **final_kwargs,
            )

            code = response.choices[0].message.content.strip()

            # Clean up code (remove markdown formatting if present)
            if code.startswith("```python"):
                code = code[9:]
            if code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]

            # Write code to file
            with open(file_path, "w") as f:
                f.write(code)

        except Exception as e:
            raise RuntimeError(f"Failed to generate code using OpenAI: {e}")

    def _import_transform(self, file_path: Path, function_name: str) -> Callable:
        """Import the generated module and return the specified function."""
        # Add the directory to sys.path temporarily
        module_dir = str(file_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)

        try:
            # Import the module
            module_name = file_path.stem
            module = __import__(module_name)

            # Get the specified function
            if not hasattr(module, function_name):
                raise AttributeError(
                    f"Generated module {module_name} does not have a '{function_name}' function"
                )

            return getattr(module, function_name)

        except Exception as e:
            raise ImportError(f"Failed to import generated transform: {e}")
        finally:
            # Remove the directory from sys.path
            if module_dir in sys.path:
                sys.path.remove(module_dir)
