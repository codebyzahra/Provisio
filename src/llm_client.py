"""
llm_client.py
=============
Shared LLM client for the Onboarding Copilot pipeline.

Reads configuration from environment variables (or a ``.env`` file via
python-dotenv) and exposes a pre-configured ``openai.OpenAI`` client plus a
convenience wrapper :func:`chat_completion`.

Environment variables
---------------------
``LLM_API_KEY``
    **Required.** API key for the LLM provider (e.g. a Groq API key).
    Raises :class:`ValueError` at import time if missing or empty.
``LLM_BASE_URL``
    Base URL of the OpenAI-compatible API endpoint.
    Default: ``"https://api.groq.com/openai/v1"``.
``LLM_MODEL_NAME``
    Model identifier to use for all completions.
    Default: ``"llama-3.1-8b-instant"``.
``LLM_TEMPERATURE``
    Sampling temperature (float, 0–2).  Default: ``0.3``.
``LLM_MAX_TOKENS``
    Maximum number of tokens in the completion.  Default: ``1024``.

Exported names
--------------
- :data:`client`       — configured ``openai.OpenAI`` instance.
- :data:`MODEL_NAME`   — resolved model identifier string.
- :data:`TEMPERATURE`  — resolved temperature float.
- :data:`MAX_TOKENS`   — resolved max-tokens int.
- :func:`chat_completion` — single-call convenience wrapper.
"""

from __future__ import annotations

import os

import openai
from dotenv import load_dotenv

# Load .env into the process environment before reading any variables.
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_api_key: str = os.environ.get("LLM_API_KEY", "")
if not _api_key:
    raise ValueError(
        "LLM_API_KEY is not set. "
        "Create a .env file in the project root and add:\n"
        "  LLM_API_KEY=<your-api-key>\n"
        "Then restart the process."
    )

_base_url: str = os.environ.get(
    "LLM_BASE_URL", "https://api.groq.com/openai/v1"
)

MODEL_NAME: str = os.environ.get("LLM_MODEL_NAME", "llama-3.1-8b-instant")
"""Resolved model identifier used for all :func:`chat_completion` calls."""

TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
"""Resolved sampling temperature used for all :func:`chat_completion` calls."""

MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", "1024"))
"""Resolved max-tokens limit used for all :func:`chat_completion` calls."""

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

client: openai.OpenAI = openai.OpenAI(api_key=_api_key, base_url=_base_url)
"""Pre-configured ``openai.OpenAI`` instance pointed at ``LLM_BASE_URL``."""

# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def chat_completion(messages: list[dict]) -> str:
    """Send *messages* to the LLM and return the assistant's reply text.

    Args:
        messages: A list of OpenAI-style message dicts, e.g.
            ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``.

    Returns:
        The text content of the first choice's message, or an empty string if
        the model returns no content.

    Raises:
        Any exception raised by the underlying ``openai`` SDK (network errors,
        rate-limit errors, authentication errors, etc.) is propagated to the
        caller without suppression — callers should implement their own
        fallback if resilience is required.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content or ""
