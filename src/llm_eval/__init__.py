"""llm_eval — async harness for evaluating LLM-generated Python code."""

from .client import LLMClient, MockClient, OpenAICompatibleClient, extract_code
from .harness import Harness, load_tasks
from .models import EvalResult, FailureMode, Task, TestCase
from .report import summarize, to_json, to_markdown
from .scorer import score

__all__ = [
    "LLMClient", "MockClient", "OpenAICompatibleClient", "extract_code",
    "Harness", "load_tasks",
    "EvalResult", "FailureMode", "Task", "TestCase",
    "summarize", "to_json", "to_markdown",
    "score",
]

__version__ = "0.1.0"
