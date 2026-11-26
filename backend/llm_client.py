"""Unified LLM client with provider-aware routing.

Routing priority for every model call:
  1. Direct provider API  — if a provider-specific key is set in the environment
                            (OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY)
  2. OpenRouter fallback  — uses OPENROUTER_API_KEY, exactly as before

Rate-limit protections for OpenRouter free tier:
  - Async semaphore limits concurrent OpenRouter free requests to 2
  - 429 responses honour the Retry-After header (fallback 10s), up to 3 retries
  - 502/503 retry after 5s, up to 2 times
  - Per-model cooldown tracking (60s after exhausting retries)

query_model_routed() is the dict-based entry point that reads
model_config["provider"] for explicit routing.
"""

import asyncio
import logging
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import httpx

from .config import (
    OPENROUTER_API_URL,
    get_model_id,
    get_openrouter_api_key,
    get_openrouter_free_models_only,
    get_provider_fallback_to_openrouter,
    get_provider_keys,
)

logger = logging.getLogger(__name__)

# ── Rate-limit concurrency control ────────────────────────────────────────────
# At most 2 concurrent OpenRouter free-tier requests
_openrouter_free_semaphore = asyncio.Semaphore(2)

# ── Per-model cooldown tracking ───────────────────────────────────────────────
# Maps model_id → timestamp when the cooldown expires
_cooldown_until: Dict[str, float] = {}

# ── Per-session usage counter for rotating selection ──────────────────────────
model_usage_counter: Counter = Counter()


def is_model_cooling_down(model_id: str) -> bool:
    """Check if a model is currently in cooldown."""
    deadline = _cooldown_until.get(model_id)
    if deadline is None:
        return False
    if time.time() >= deadline:
        # Cooldown expired — remove entry
        _cooldown_until.pop(model_id, None)
        logger.info("[COOLDOWN EXPIRED] %s is eligible again", model_id)
        return False
    return True


def set_model_cooldown(model_id: str, seconds: float = 60.0) -> None:
    """Mark a model as cooling down for the given duration."""
    _cooldown_until[model_id] = time.time() + seconds
    logger.warning(
        "[COOLDOWN] %s cooling down for %.0fs after failed attempts",
        model_id, seconds,
    )


# ── Provider detection ────────────────────────────────────────────────────────

# Explicit prefix → provider table (OpenRouter uses "provider/model" strings)
_PREFIX_MAP: Dict[str, str] = {
    "openai":     "openai",
    "anthropic":  "anthropic",
    "google":     "google",
    "groq":       "groq",
    "gemini":     "google",    # some older OpenRouter slugs use "gemini/..."
}

# Keyword fragments used when the model name has no "/" prefix
_KEYWORD_MAP: list[tuple[tuple[str, ...], str]] = [
    (("gpt-", "o1-", "o3-", "o4-", "davinci", "text-embedding"), "openai"),
    (("claude",),                                                  "anthropic"),
    (("gemini", "bison", "palm"),                                  "google"),
    (("llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"), "groq"),
]


def infer_provider(model_name: str) -> str:
    """Return the provider for *model_name*.

    For OpenRouter-style "provider/model" strings the prefix is authoritative.
    For bare model names keyword matching is used.

    Returns one of: "openai" | "anthropic" | "google" | "groq" | "openrouter"
    The sentinel value "openrouter" means "route via OpenRouter".
    """
    lower = model_name.lower()

    # OpenRouter free-tier slugs are provider-routed marketplace ids, not native
    # provider model names. Keep them on OpenRouter even if they start with
    # "google/", "openai/", etc.
    if ":free" in lower:
        return "openrouter"

    # OpenRouter uses "provider/model" — honour the explicit prefix
    if "/" in lower:
        prefix = lower.split("/")[0]
        if prefix in _PREFIX_MAP:
            return _PREFIX_MAP[prefix]
        # Unknown prefix (e.g. "x-ai/grok-4", "mistralai/...") → OpenRouter
        return "openrouter"

    # Bare model name — fall back to keyword matching
    for keywords, provider in _KEYWORD_MAP:
        if any(k in lower for k in keywords):
            return provider

    return "openrouter"


# ── Direct provider calls ─────────────────────────────────────────────────────

def _strip_prefix(model: str) -> str:
    """Return bare model name, stripping any "provider/" prefix."""
    return model.split("/", 1)[-1] if "/" in model else model


async def _query_openai(
    model: str,
    messages: List[Dict[str, str]],
    api_key: str,
    *,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Call the OpenAI Chat Completions API directly."""
    try:
        from openai import AsyncOpenAI  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "openai package is not installed. "
            "Install it with: pip install openai"
        ) from exc

    bare_model = _strip_prefix(model)
    client = AsyncOpenAI(api_key=api_key)
    kwargs: Dict[str, Any] = {"model": bare_model, "messages": messages}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    completion = await client.chat.completions.create(**kwargs)
    msg = completion.choices[0].message
    return {
        "content": msg.content,
        "reasoning_details": None,
    }


async def _query_anthropic(
    model: str,
    messages: List[Dict[str, str]],
    api_key: str,
    *,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Call the Anthropic Messages API directly."""
    try:
        import anthropic as anthropic_sdk  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "anthropic package is not installed. "
            "Install it with: pip install anthropic"
        ) from exc

    bare_model = _strip_prefix(model)
    client = anthropic_sdk.AsyncAnthropic(api_key=api_key)

    # Separate optional system message from conversation turns
    system_content: Optional[str] = None
    conv_messages: List[Dict[str, str]] = []
    for m in messages:
        if m["role"] == "system":
            system_content = m["content"]
        else:
            conv_messages.append(m)

    kwargs: Dict[str, Any] = {
        "model": bare_model,
        "max_tokens": max_tokens or 8192,
        "messages": conv_messages,
    }
    if system_content:
        kwargs["system"] = system_content

    response = await client.messages.create(**kwargs)
    return {
        "content": response.content[0].text,
        "reasoning_details": None,
    }


async def _query_google(
    model: str,
    messages: List[Dict[str, str]],
    api_key: str,
    *,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Call the Google Generative AI API directly."""
    try:
        import google.generativeai as genai  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai package is not installed. "
            "Install it with: pip install google-generativeai"
        ) from exc

    bare_model = _strip_prefix(model)
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(bare_model)

    gen_config = {}
    if max_tokens is not None:
        gen_config["max_output_tokens"] = max_tokens

    # For the current single-turn usage (always one user message), send directly.
    if len(messages) == 1 and messages[0]["role"] == "user":
        response = await gemini_model.generate_content_async(
            messages[0]["content"],
            generation_config=gen_config if gen_config else None,
        )
    else:
        # Convert OpenAI-style message list to Gemini chat history format
        history = []
        for m in messages[:-1]:
            role = "user" if m["role"] == "user" else "model"
            history.append({"role": role, "parts": [m["content"]]})
        chat = gemini_model.start_chat(history=history)
        last_msg = messages[-1]["content"]
        response = await chat.send_message_async(last_msg)

    return {
        "content": response.text,
        "reasoning_details": None,
    }


async def _query_groq(
    model: str,
    messages: List[Dict[str, str]],
    api_key: str,
    *,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Call the Groq Chat Completions API directly."""
    try:
        from groq import AsyncGroq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "groq package is not installed. "
            "Install it with: pip install groq"
        ) from exc

    bare_model = _strip_prefix(model)
    client = AsyncGroq(api_key=api_key)
    kwargs: Dict[str, Any] = {"model": bare_model, "messages": messages}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    completion = await client.chat.completions.create(**kwargs)
    msg = completion.choices[0].message
    return {
        "content": msg.content,
        "reasoning_details": None,
    }


# ── OpenRouter (with rate-limit-aware retry) ──────────────────────────────────

_FREE_MODEL_FALLBACKS: Dict[str, str] = {
    "openai": "meta-llama/llama-3.3-70b-instruct:free",
    "anthropic": "qwen/qwen3-32b:free",
    "google": "google/gemma-3-27b-it:free",
    "groq": "deepseek/deepseek-r1-0528:free",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
}


def _resolve_openrouter_model(model: str) -> str:
    """Return an OpenRouter-safe model identifier under free-only policy."""
    if not get_openrouter_free_models_only():
        return model

    if model.endswith(":free"):
        return model

    provider = infer_provider(model)
    fallback = _FREE_MODEL_FALLBACKS.get(provider, _FREE_MODEL_FALLBACKS["openrouter"])
    logger.warning(
        "Model %s is not marked free for OpenRouter. Using fallback free model %s.",
        model,
        fallback,
    )
    return fallback


async def _query_via_openrouter(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 30.0,
    *,
    max_tokens: Optional[int] = None,
    is_free_tier: bool = False,
) -> Optional[Dict[str, Any]]:
    """Send *model* request through OpenRouter with rate-limit-aware retry.

    Retry policy:
      - 429: wait Retry-After header seconds (fallback 10s), up to 3 retries
      - 502/503: wait 5s, up to 2 retries
    Free-tier requests are serialized through a semaphore (max 2 concurrent).
    """
    openrouter_api_key = get_openrouter_api_key()
    if not openrouter_api_key:
        logger.error("OPENROUTER_API_KEY is not configured")
        return None

    routed_model = _resolve_openrouter_model(model)

    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {"model": routed_model, "messages": messages}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    # Retry budgets per error type
    max_429_retries = 3
    max_5xx_retries = 2
    retries_429 = 0
    retries_5xx = 0

    async def _do_request() -> Optional[Dict[str, Any]]:
        nonlocal retries_429, retries_5xx

        while True:
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        OPENROUTER_API_URL,
                        headers=headers,
                        json=payload,
                    )

                    # ── 429 Too Many Requests ──
                    if response.status_code == 429:
                        retries_429 += 1
                        if retries_429 > max_429_retries:
                            logger.error(
                                "[RETRY EXHAUSTED] %s: 429 after %d retries",
                                model, max_429_retries,
                            )
                            return None
                        # Honour Retry-After header if present
                        retry_after = response.headers.get("retry-after")
                        if retry_after:
                            try:
                                delay = float(retry_after)
                            except ValueError:
                                delay = 10.0
                        else:
                            delay = 10.0
                        logger.warning(
                            "[RETRY] %s: 429 Too Many Requests (attempt %d/%d). "
                            "Waiting %.1fs (Retry-After: %s)...",
                            model, retries_429, max_429_retries, delay,
                            retry_after or "not set",
                        )
                        await asyncio.sleep(delay)
                        continue

                    # ── 502/503 Server Errors ──
                    if response.status_code in (502, 503):
                        retries_5xx += 1
                        if retries_5xx > max_5xx_retries:
                            logger.error(
                                "[RETRY EXHAUSTED] %s: %d after %d retries",
                                model, response.status_code, max_5xx_retries,
                            )
                            return None
                        logger.warning(
                            "[RETRY] %s: %d (attempt %d/%d). Waiting 5s...",
                            model, response.status_code,
                            retries_5xx, max_5xx_retries,
                        )
                        await asyncio.sleep(5.0)
                        continue

                    response.raise_for_status()
                    data = response.json()
                    message = data["choices"][0]["message"]
                    return {
                        "content": message.get("content"),
                        "reasoning_details": message.get("reasoning_details"),
                    }

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status == 429 and retries_429 < max_429_retries:
                    retries_429 += 1
                    retry_after = exc.response.headers.get("retry-after")
                    delay = 10.0
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            pass
                    logger.warning(
                        "[RETRY] %s: 429 (attempt %d/%d). Waiting %.1fs...",
                        model, retries_429, max_429_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                if status in (502, 503) and retries_5xx < max_5xx_retries:
                    retries_5xx += 1
                    logger.warning(
                        "[RETRY] %s: %d (attempt %d/%d). Waiting 5s...",
                        model, status, retries_5xx, max_5xx_retries,
                    )
                    await asyncio.sleep(5.0)
                    continue
                logger.error("OpenRouter error for %s (routed as %s): %s", model, routed_model, exc)
                return None

            except Exception as exc:
                logger.error("OpenRouter error for %s (routed as %s): %s", model, routed_model, exc)
                return None

    # Free-tier: acquire semaphore to limit concurrency
    if is_free_tier:
        async with _openrouter_free_semaphore:
            return await _do_request()
    else:
        return await _do_request()


# ── Dict-based routing entry point ────────────────────────────────────────────

async def query_model_routed(
    model_config: Dict[str, str],
    messages: List[Dict[str, str]],
    *,
    max_tokens: Optional[int] = None,
    timeout: float = 30.0,
) -> Optional[Dict[str, Any]]:
    """Route a query based on model_config["provider"] with per-model timeout.

    Args:
        model_config: Dict with keys: id, provider, label, tier
        messages: Chat messages
        max_tokens: Optional token limit
        timeout: Per-model timeout in seconds (default 30s). Returns None on timeout.

    Returns:
        {'content': str, 'reasoning_details': ...} or None on failure/timeout.
    """
    model_id = model_config["id"]
    provider = model_config.get("provider", "openrouter")
    is_free = model_config.get("tier", "free") == "free"

    # Skip models in cooldown
    if is_model_cooling_down(model_id):
        logger.info("[SKIP] %s is in cooldown, skipping", model_id)
        return None

    try:
        result = await asyncio.wait_for(
            _query_model_routed_inner(
                model_id, provider, messages,
                max_tokens=max_tokens, is_free_tier=is_free,
            ),
            timeout=timeout,
        )
        if result is None and provider == "openrouter" and is_free:
            # All retries exhausted — put model in cooldown
            set_model_cooldown(model_id, 60.0)
        return result
    except asyncio.TimeoutError:
        logger.warning("Timeout (%.0fs) for model %s (%s). Returning None.", timeout, model_id, provider)
        if provider == "openrouter" and is_free:
            set_model_cooldown(model_id, 60.0)
        return None
    except Exception as exc:
        logger.error("Unexpected error querying %s via %s: %s", model_id, provider, exc)
        if provider == "openrouter" and is_free:
            set_model_cooldown(model_id, 60.0)
        return None


async def _query_model_routed_inner(
    model_id: str,
    provider: str,
    messages: List[Dict[str, str]],
    *,
    max_tokens: Optional[int] = None,
    is_free_tier: bool = False,
) -> Optional[Dict[str, Any]]:
    """Inner dispatch: route to the correct provider SDK."""
    keys = get_provider_keys()

    if provider == "groq":
        api_key = keys.get("groq")
        if not api_key:
            logger.error("GROQ_API_KEY is not configured but model %s requires Groq", model_id)
            return None
        return await _query_groq(model_id, messages, api_key, max_tokens=max_tokens)

    if provider == "openai":
        api_key = keys.get("openai")
        if api_key:
            try:
                return await _query_openai(model_id, messages, api_key, max_tokens=max_tokens)
            except Exception as exc:
                logger.warning("Direct OpenAI call failed for %s: %s. Falling back to OpenRouter.", model_id, exc)

    if provider == "anthropic":
        api_key = keys.get("anthropic")
        if api_key:
            try:
                return await _query_anthropic(model_id, messages, api_key, max_tokens=max_tokens)
            except Exception as exc:
                logger.warning("Direct Anthropic call failed for %s: %s. Falling back to OpenRouter.", model_id, exc)

    if provider == "google":
        api_key = keys.get("google")
        if api_key:
            try:
                return await _query_google(model_id, messages, api_key, max_tokens=max_tokens)
            except Exception as exc:
                logger.warning("Direct Google call failed for %s: %s. Falling back to OpenRouter.", model_id, exc)

    # Default / fallback: OpenRouter
    return await _query_via_openrouter(
        model_id, messages, timeout=30.0, max_tokens=max_tokens, is_free_tier=is_free_tier,
    )


# ── Legacy unified entry point (preserved for backward compatibility) ─────────

async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
) -> Optional[Dict[str, Any]]:
    """Route a query to the appropriate provider and return a normalised response.

    This is the legacy string-based entry point, kept for backward compatibility
    (e.g. generate_conversation_title). For new code, prefer query_model_routed().
    """
    provider = infer_provider(model)
    keys = get_provider_keys()
    direct_key = keys.get(provider)
    fallback_to_openrouter = get_provider_fallback_to_openrouter()

    # ── Direct provider path ──────────────────────────────────────────────────
    if provider != "openrouter" and direct_key:
        logger.debug("Routing %s via direct %s API", model, provider)
        try:
            if provider == "openai":
                return await _query_openai(model, messages, direct_key)
            elif provider == "anthropic":
                return await _query_anthropic(model, messages, direct_key)
            elif provider == "google":
                return await _query_google(model, messages, direct_key)
            elif provider == "groq":
                return await _query_groq(model, messages, direct_key)
        except Exception as exc:
            if fallback_to_openrouter:
                logger.warning(
                    "Direct %s call failed for model %s (%s). "
                    "Falling back to OpenRouter.",
                    provider, model, exc,
                )
            else:
                logger.error(
                    "Direct %s call failed for model %s (%s). "
                    "Fallback disabled — returning None.",
                    provider, model, exc,
                )
                return None

    # ── OpenRouter path (default / fallback) ──────────────────────────────────
    logger.debug("Routing %s via OpenRouter", model)
    return await _query_via_openrouter(model, messages, timeout=timeout)


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Query multiple models in parallel using the unified routing layer.

    Returns a dict mapping each model identifier to its response (or None).
    """
    tasks = [query_model(model, messages) for model in models]
    responses = await asyncio.gather(*tasks)
    return {model: response for model, response in zip(models, responses)}
