"""OpenRouter API client — public interface preserved for backward compatibility.

All routing logic now lives in llm_client.py.  This module re-exports the
functions that the rest of the codebase imports so that no other file needs to
change.
"""

from typing import Any, Dict, List, Optional

# Re-export the unified routing functions under their original names.
# query_model() will use a direct provider key when available and fall back to
# OpenRouter otherwise.  query_models_parallel() runs them concurrently.
# query_model_routed() is the new dict-based entry point.
from .llm_client import query_model, query_models_parallel, query_model_routed

__all__ = ["query_model", "query_models_parallel", "query_model_routed"]
