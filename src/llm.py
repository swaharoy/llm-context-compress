"""LLM access: one fixed prompt, a Gemini client, a fake client and a disk cache (specs.md §7).

Every experiment, baseline and Bloom alike, builds its prompt with
:func:`build_prompt` and sends it through an :class:`LLMClient`. The two can then
differ only in the context they pass.

* :class:`GeminiClient`: the real model. ``google-genai`` is imported lazily, so
  tests and offline runs never need network access or an API key.
* :class:`FakeLLM`: deterministic stand-in for tests and offline dry runs.
* :class:`CachedLLM`: wraps any client, storing each response on disk keyed by
  model, generation config and prompt. Reruns are then free and give identical results.

Depends on: the standard library (plus ``google-genai`` when :class:`GeminiClient` is used).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

MODEL = "gemini-3.5-flash-lite"
# Upper bound used by the pre-flight check (§10.1). Verify it against the model card before reporting.
MAX_CONTEXT_TOKENS = 1_048_576

SYSTEM_INSTRUCTION = (
    "Answer the question using only the context provided. "
    "Reply with only the answer itself: no explanation and no full sentence."
)
PROMPT_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer type: {answer_type}; unit: {unit}\n"
    "Answer:"
)


def build_prompt(context: str, question: str, answer_type: str, unit: str | None) -> str:
    """Fill :data:`PROMPT_TEMPLATE`. ``answer_type`` and ``unit`` come from the questions file."""
    return PROMPT_TEMPLATE.format(
        context=context, question=question, answer_type=answer_type, unit=unit or "none"
    )


@dataclass(frozen=True)
class LLMResponse:
    """One model answer and its usage.

    Attributes:
        text: The model's answer, stripped.
        prompt_tokens: Input tokens as reported by the API, including the system instruction.
        output_tokens: Output tokens reported by the API.
        latency_s: Wall time of the successful request only.
        retries: Failed attempts before the success.
        cached: True when replayed from disk. Its latency is then not a fresh observation.
        model_version: The model version that actually served the request.
    """

    text: str
    prompt_tokens: int
    output_tokens: int
    latency_s: float
    retries: int = 0
    cached: bool = False
    model_version: str = ""


class LLMClient(Protocol):
    """A model that answers prompts."""

    model: str

    def generation_config(self) -> dict:
        """Return every setting that affects the answer. It is part of the cache key."""
        ...

    def generate(self, prompt: str) -> LLMResponse: ...


class GeminiClient:
    """Calls Gemini with fixed generation settings and retries transient errors.

    The API key is read from ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` by the SDK.
    """

    RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        model: str = MODEL,
        temperature: float = 0.0,
        seed: int = 580,
        max_output_tokens: int = 64,
        max_retries: int = 5,
    ):
        from google import genai  # lazy: only real runs need the SDK and network

        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self._client = genai.Client()

    def generation_config(self) -> dict:
        return {
            "system_instruction": SYSTEM_INSTRUCTION,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_output_tokens": self.max_output_tokens,
        }

    def generate(self, prompt: str) -> LLMResponse:
        from google.genai import errors, types

        config = types.GenerateContentConfig(**self.generation_config())
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
            except errors.APIError as error:
                if error.code not in self.RETRYABLE_CODES or attempt == self.max_retries:
                    raise
                time.sleep(2 ** attempt)
                continue
            latency = time.perf_counter() - start
            usage = response.usage_metadata
            return LLMResponse(
                text=(response.text or "").strip(),
                prompt_tokens=usage.prompt_token_count or 0,
                output_tokens=usage.candidates_token_count or 0,
                latency_s=latency,
                retries=attempt,
                model_version=response.model_version or self.model,
            )
        raise RuntimeError("unreachable")

    def count_tokens(self, text: str) -> int:
        """Count tokens with the API. Used by the baseline pre-flight check."""
        return self._client.models.count_tokens(model=self.model, contents=text).total_tokens


class FakeLLM:
    """Deterministic offline stand-in: answers with the first number in the context.

    It checks the plumbing (prompts, records, grading, resume), never accuracy.
    Its token counts use the same ``ceil(chars/4)`` estimate as retrieval.
    """

    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def generation_config(self) -> dict:
        return {"system_instruction": SYSTEM_INSTRUCTION}

    def generate(self, prompt: str) -> LLMResponse:
        self.calls += 1
        context = prompt.split("\n\nQuestion:")[0]
        number = re.search(r"\d+(?:\.\d+)?", context)
        answer = number.group(0) if number else "unknown"
        return LLMResponse(answer, -(-len(prompt) // 4), 1, 0.0, model_version="fake")


class CachedLLM:
    """Wraps a client with a JSON-file cache in ``cache_dir``.

    The key is ``sha256(model | generation config | prompt)``. Identical prompts,
    such as those from the row-major and bit-sliced indexes, are answered once.
    With ``enabled=False`` every call goes to the inner client, which is how
    fresh latency is measured, and the cache is still refreshed.
    """

    def __init__(self, inner: LLMClient, cache_dir: Path, enabled: bool = True):
        self.inner = inner
        self.model = inner.model
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generation_config(self) -> dict:
        return self.inner.generation_config()

    def key(self, prompt: str) -> str:
        payload = json.dumps(
            {"model": self.model, "config": self.generation_config(), "prompt": prompt},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def generate(self, prompt: str) -> LLMResponse:
        path = self.cache_dir / f"{self.key(prompt)}.json"
        if self.enabled and path.exists():
            stored = json.loads(path.read_text())
            return replace(LLMResponse(**stored), cached=True)
        response = self.inner.generate(prompt)
        path.write_text(json.dumps(asdict(response)))
        return response
