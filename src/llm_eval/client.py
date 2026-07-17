"""LLM client abstraction.

The harness talks to any OpenAI-compatible chat-completions endpoint
(OpenAI, Anthropic via gateway, Ollama, vLLM, LM Studio...). A MockClient
is included so the pipeline is fully testable offline.
"""

from __future__ import annotations

import abc
import os
import re

import httpx

SYSTEM_PROMPT = (
    "You are an expert Python engineer. Solve the task with a single, "
    "self-contained Python function. Reply with ONLY a Python code block. "
    "Do not include usage examples, prints, or explanations."
)

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull Python source out of a model reply.

    Models often wrap code in markdown fences or add prose around it;
    grading raw replies without normalization inflates failure counts.
    """
    match = _CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


class LLMClient(abc.ABC):
    """Minimal interface the harness needs from a model."""

    name: str = "unknown"

    @abc.abstractmethod
    async def generate(self, prompt: str) -> str:
        """Return the model's raw reply for a coding prompt."""


class OpenAICompatibleClient(LLMClient):
    """Async client for any OpenAI-compatible /chat/completions API."""

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout_s: float = 60.0,
        temperature: float = 0.0,
    ) -> None:
        self.name = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self._timeout_s = timeout_s
        self._temperature = temperature

    async def generate(self, prompt: str) -> str:
        payload = {
            "model": self.name,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        # Local endpoints (Ollama, vLLM) need no key; an empty
        # "Bearer " value is an illegal HTTP header, so omit it entirely.
        headers = (
            {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        )
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]


class MockClient(LLMClient):
    """Deterministic fake model for offline tests and CI.

    Responses are keyed by a substring of the prompt; unmatched prompts
    get the default reply.
    """

    name = "mock-model"

    def __init__(self, responses: dict[str, str] | None = None, default: str = "") -> None:
        self._responses = responses or {}
        self._default = default
        self.calls: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        for key, reply in self._responses.items():
            if key in prompt:
                return reply
        return self._default
