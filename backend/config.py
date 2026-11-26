"""Configuration for the LLM Council."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"

# ---------------------------------------------------------------------------
# Model configuration — each model is a dict with:
#   id       : model identifier (bare name for Groq, OpenRouter slug otherwise)
#   provider : "groq" | "openrouter"
#   label    : human-readable display name
#   tier     : "free" | "paid"
# ---------------------------------------------------------------------------

COUNCIL_MODELS: List[Dict[str, str]] = [
    # Groq (fast, dedicated — not subject to OpenRouter rate limits)
    {"id": "llama-3.1-8b-instant", "provider": "groq", "label": "Llama 3.1 8B", "tier": "free"},
    {"id": "llama-3.3-70b-versatile", "provider": "groq", "label": "Llama 3.3 70B", "tier": "free"},

    # OpenRouter Free
    {"id": "nvidia/llama-3.1-nemotron-70b-instruct:free", "provider": "openrouter", "label": "Nemotron 70B", "tier": "free"},
    {"id": "mistralai/mistral-7b-instruct:free", "provider": "openrouter", "label": "Mistral 7B", "tier": "free"},
    {"id": "google/gemini-2.0-flash-001:free", "provider": "openrouter", "label": "Gemini 2.0 Flash", "tier": "free"},
    {"id": "meta-llama/llama-3.3-70b-instruct:free", "provider": "openrouter", "label": "Llama 3.3 70B (OR)", "tier": "free"},
    {"id": "deepseek/deepseek-r1:free", "provider": "openrouter", "label": "DeepSeek R1", "tier": "free"},
    {"id": "qwen/qwen3-235b-a22b:free", "provider": "openrouter", "label": "Qwen3 235B", "tier": "free"},
]

# Number of models to select per request from the pool
COUNCIL_SELECT_COUNT = 4

CHAIRMAN_MODEL: Dict[str, str] = {
    "id": "llama-3.3-70b-versatile",
    "provider": "groq",
    "label": "Llama 3.3 70B Chairman",
    "tier": "free",
}

_runtime_overrides: Dict[str, Any] = {}
_config_cache: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_model_id(model_config: Union[Dict[str, str], str]) -> str:
    """Extract the model ID string from a model config dict or plain string."""
    if isinstance(model_config, dict):
        return model_config["id"]
    return model_config


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _extract_key(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        nested = value.get("api_key") or value.get("key")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def _get_config_path() -> Optional[Path]:
    configured = _runtime_overrides.get("config_path") or os.getenv("COUNCIL_CONFIG_FILE")
    if not configured:
        for candidate in ("council_config.json", "config.yaml", "config.yml"):
            candidate_path = Path(candidate)
            if candidate_path.exists():
                return candidate_path
        return None
    path = Path(configured)
    return path if path.exists() else None


def _load_config_file() -> Dict[str, Any]:
    global _config_cache

    if _config_cache is not None:
        return _config_cache

    path = _get_config_path()
    if path is None:
        _config_cache = {}
        return _config_cache

    try:
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()

        if suffix == ".json":
            _config_cache = json.loads(text)
            return _config_cache

        if suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError:
                logger.warning(
                    "Config file %s is YAML but PyYAML is not installed. "
                    "Install with: pip install pyyaml",
                    path,
                )
                _config_cache = {}
                return _config_cache
            parsed = yaml.safe_load(text)
            _config_cache = parsed if isinstance(parsed, dict) else {}
            return _config_cache

        logger.warning("Unsupported config file type for %s. Use .json or .yaml/.yml", path)
        _config_cache = {}
        return _config_cache
    except Exception as exc:
        logger.warning("Failed to load config file %s: %s", path, exc)
        _config_cache = {}
        return _config_cache


def set_runtime_overrides(
    *,
    openai_key: Optional[str] = None,
    anthropic_key: Optional[str] = None,
    google_key: Optional[str] = None,
    groq_key: Optional[str] = None,
    openrouter_key: Optional[str] = None,
    provider_fallback_to_openrouter: Optional[bool] = None,
    config_path: Optional[str] = None,
) -> None:
    """Set runtime overrides (typically from CLI flags)."""
    global _config_cache

    if openai_key is not None:
        _runtime_overrides["openai_key"] = openai_key
    if anthropic_key is not None:
        _runtime_overrides["anthropic_key"] = anthropic_key
    if google_key is not None:
        _runtime_overrides["google_key"] = google_key
    if groq_key is not None:
        _runtime_overrides["groq_key"] = groq_key
    if openrouter_key is not None:
        _runtime_overrides["openrouter_key"] = openrouter_key
    if provider_fallback_to_openrouter is not None:
        _runtime_overrides["provider_fallback_to_openrouter"] = provider_fallback_to_openrouter
    if config_path is not None:
        _runtime_overrides["config_path"] = config_path
        _config_cache = None


def get_provider_keys() -> Dict[str, Optional[str]]:
    """Get provider keys using precedence: CLI > config file > environment."""
    cfg = _load_config_file()
    providers = cfg.get("providers") if isinstance(cfg, dict) else {}
    providers = providers if isinstance(providers, dict) else {}

    env_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "google": os.getenv("GOOGLE_API_KEY"),
        "groq": os.getenv("GROQ_API_KEY"),
        "openrouter": os.getenv("OPENROUTER_API_KEY"),
    }

    cfg_keys = {
        "openai": _extract_key(providers.get("openai")),
        "anthropic": _extract_key(providers.get("anthropic")),
        "google": _extract_key(providers.get("google")),
        "groq": _extract_key(providers.get("groq")),
        "openrouter": _extract_key(providers.get("openrouter")),
    }

    cli_keys = {
        "openai": _runtime_overrides.get("openai_key"),
        "anthropic": _runtime_overrides.get("anthropic_key"),
        "google": _runtime_overrides.get("google_key"),
        "groq": _runtime_overrides.get("groq_key"),
        "openrouter": _runtime_overrides.get("openrouter_key"),
    }

    merged: Dict[str, Optional[str]] = {}
    for provider in ("openai", "anthropic", "google", "groq", "openrouter"):
        merged[provider] = cli_keys.get(provider) or cfg_keys.get(provider) or env_keys.get(provider)

    return merged


def get_openrouter_api_key() -> Optional[str]:
    return get_provider_keys().get("openrouter")


def get_provider_fallback_to_openrouter() -> bool:
    """Get fallback toggle using precedence: CLI > config file > env > default(true)."""
    if "provider_fallback_to_openrouter" in _runtime_overrides:
        return bool(_runtime_overrides["provider_fallback_to_openrouter"])

    cfg = _load_config_file()
    providers = cfg.get("providers") if isinstance(cfg, dict) else {}
    providers = providers if isinstance(providers, dict) else {}
    cfg_value = providers.get("fallback_to_openrouter")
    if cfg_value is not None:
        return bool(cfg_value)

    env_value = os.getenv("PROVIDER_FALLBACK_TO_OPENROUTER")
    if env_value is None:
        return True
    return _truthy(env_value)


def get_openrouter_free_models_only() -> bool:
    """Whether OpenRouter routes should be constrained to free models only."""
    cfg = _load_config_file()

    openrouter_cfg = cfg.get("openrouter") if isinstance(cfg, dict) else {}
    if isinstance(openrouter_cfg, dict) and "free_models_only" in openrouter_cfg:
        return bool(openrouter_cfg["free_models_only"])

    env_value = os.getenv("OPENROUTER_FREE_MODELS_ONLY")
    if env_value is None:
        return True
    return _truthy(env_value)


def get_council_models() -> List[Dict[str, str]]:
    """Return the list of council model config dicts."""
    cfg = _load_config_file()
    configured = cfg.get("council_models") if isinstance(cfg, dict) else None
    if isinstance(configured, list) and configured:
        # Support both dict-style and plain-string entries from config files
        result = []
        for entry in configured:
            if isinstance(entry, dict) and "id" in entry:
                result.append(entry)
            elif isinstance(entry, str):
                result.append({
                    "id": entry,
                    "provider": "openrouter",
                    "label": entry,
                    "tier": "free",
                })
        return result if result else COUNCIL_MODELS
    return COUNCIL_MODELS


def get_chairman_model() -> Dict[str, str]:
    """Return the chairman model config dict."""
    cfg = _load_config_file()
    configured = cfg.get("chairman_model") if isinstance(cfg, dict) else None
    if isinstance(configured, dict) and "id" in configured:
        return configured
    if isinstance(configured, str) and configured.strip():
        return {
            "id": configured.strip(),
            "provider": "openrouter",
            "label": configured.strip(),
            "tier": "free",
        }
    return CHAIRMAN_MODEL
