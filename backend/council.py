"""3-stage LLM Council orchestration."""

import asyncio
import logging
import re
from typing import List, Dict, Any, Tuple, Optional

from .llm_client import query_model, query_model_routed
from .config import get_council_models, get_chairman_model, get_model_id

logger = logging.getLogger(__name__)

# Stagger delay (in seconds) between free-tier model requests
_FREE_TIER_STAGGER_DELAY = 1.5


def _get_voting_models() -> List[Dict[str, str]]:
    """Return council models excluding the chairman for fair voting."""
    chairman = get_chairman_model()
    chairman_id = get_model_id(chairman)
    models = [m for m in get_council_models() if get_model_id(m) != chairman_id]
    # If config accidentally only includes the chairman, keep original list so
    # the system remains usable instead of hard-failing.
    return models or get_council_models()


async def _query_models_sequential(
    models: List[Dict[str, str]],
    messages: List[Dict[str, str]],
    *,
    max_tokens: Optional[int] = None,
    max_tokens_groq: Optional[int] = None,
) -> Dict[str, Any | None]:
    """Query models one-by-one with stagger for free-tier models.

    Args:
        models: List of model config dicts
        messages: Chat messages to send
        max_tokens: Default max_tokens for non-Groq models (None = no limit)
        max_tokens_groq: Override max_tokens for Groq models (None = use max_tokens)
    """
    results: Dict[str, Any | None] = {}
    for i, model_config in enumerate(models):
        model_id = get_model_id(model_config)
        is_free = model_config.get("tier", "free") == "free"

        # Determine max_tokens for this model
        if model_config.get("provider") == "groq" and max_tokens_groq is not None:
            mt = max_tokens_groq
        else:
            mt = max_tokens

        # Stagger free-tier models (skip delay for the first model)
        if i > 0 and is_free:
            await asyncio.sleep(_FREE_TIER_STAGGER_DELAY)

        results[model_id] = await query_model_routed(
            model_config, messages, max_tokens=mt, timeout=30.0,
        )
    return results


async def stage1_collect_responses(user_query: str) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.

    Args:
        user_query: The user's question

    Returns:
        List of dicts with 'model' and 'response' keys
    """
    messages = [{"role": "user", "content": user_query}]
    voting_models = _get_voting_models()

    # Query one-by-one with stagger for better stability on free tiers.
    # Groq models capped at 1024 tokens for Stage 1.
    responses = await _query_models_sequential(
        voting_models, messages, max_tokens_groq=1024,
    )

    # Format results
    stage1_results = []
    for model_id, response in responses.items():
        if response is not None:  # Only include successful responses
            stage1_results.append({
                "model": model_id,
                "response": response.get('content', '')
            })

    # Warn if more than half the council failed
    total = len(voting_models)
    succeeded = len(stage1_results)
    failed = total - succeeded
    if failed > total / 2:
        logger.warning(
            "Stage 1: %d/%d models failed (more than half). Proceeding with %d survivors.",
            failed, total, succeeded,
        )

    return stage1_results


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks ALL anonymized responses.
    Anonymization (Response A/B/C) is the bias protection — models cannot
    see whose response is whose, so they evaluate on merit alone.

    Args:
        user_query: The original user query
        stage1_results: Results from Stage 1

    Returns:
        Tuple of (rankings list, label_to_model mapping)
    """
    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name (model ID string)
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }

    # Build the ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])

    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]

    # Build model config dicts for the evaluator models
    # (models that actually produced Stage 1 responses)
    all_council = get_council_models()
    council_by_id = {get_model_id(m): m for m in all_council}

    evaluator_configs = []
    for result in stage1_results:
        model_id = result["model"]
        if model_id in council_by_id:
            evaluator_configs.append(council_by_id[model_id])
        else:
            # Fallback: build a minimal config for unknown model IDs
            evaluator_configs.append({
                "id": model_id,
                "provider": "openrouter",
                "label": model_id,
                "tier": "free",
            })

    # Groq models capped at 512 tokens for Stage 2
    responses = await _query_models_sequential(
        evaluator_configs, messages, max_tokens_groq=512,
    )

    # Format results
    stage2_results = []
    for model_id, response in responses.items():
        if response is not None:
            full_text = response.get('content', '')
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model_id,
                "ranking": full_text,
                "parsed_ranking": parsed
            })

    return stage2_results, label_to_model


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    aggregate_rankings: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response using blind, structured synthesis.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2
        aggregate_rankings: Sorted aggregate ranking data

    Returns:
        Dict with 'model' and 'response' keys
    """
    # Build fully anonymized Stage 1 context — strip ALL model names
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }
    model_to_label = {model: label for label, model in label_to_model.items()}

    stage1_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])

    # Anonymize Stage 2 evaluations — strip any model names that may appear
    stage2_text = "\n\n".join([
        f"Evaluation {idx + 1}:\n{result['ranking']}"
        for idx, result in enumerate(stage2_results)
    ])

    # Build aggregate rankings text using ONLY anonymous labels
    aggregate_text = "No aggregate ranking available."
    if aggregate_rankings:
        aggregate_lines = []
        for item in aggregate_rankings:
            model = item.get("model")
            label = model_to_label.get(model, "Unknown")
            avg_rank = item.get("average_rank")
            votes = item.get("rankings_count")
            aggregate_lines.append(
                f"{label}: average_rank={avg_rank} ({votes} votes) — lower rank number = rated higher by peers"
            )
        aggregate_text = "\n".join(aggregate_lines)

    chairman_config = get_chairman_model()
    chairman_model_id = get_model_id(chairman_config)

    chairman_prompt = f"""You are synthesizing a final answer from multiple anonymous responses to a user's question. You have access to the responses, peer evaluations, and aggregate rankings.

Original Question: {user_query}

RESPONSES (anonymous labels only — no model names):
{stage1_text}

PEER EVALUATIONS (anonymous):
{stage2_text}

AGGREGATE PEER RANKINGS (lower average_rank = rated higher by peers):
{aggregate_text}

YOUR TASK — produce a structured synthesis:

1. CONSENSUS: Identify the key points where most responses agree.
2. DISAGREEMENTS & UNIQUE INSIGHTS: Note any meaningful differences or valuable unique contributions from individual responses.
3. FINAL ANSWER: Produce a single, authoritative answer that:
   - Weights higher-ranked responses proportionally (those with lower average_rank numbers were rated more highly by peers)
   - Integrates consensus points and the strongest unique insights
   - Resolves contradictions in favor of higher-ranked responses

STRICT RULES:
- Do NOT copy any single response verbatim. Synthesize across all of them.
- Do NOT begin with praise or meta-commentary. Start directly with the answer.
- Do NOT write phrases like "As the chairman...", "After reviewing...", "The council has determined...", "Based on the responses...", or any similar preamble.
- Output clean markdown only. No labels, no internal ranking metadata, no mention of "Response A" etc.
- The user should read your output as a direct, standalone answer to their question.

Write the final synthesized answer now:"""

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model with retry (try twice on failure)
    response = await query_model_routed(chairman_config, messages, timeout=30.0)
    if response is None:
        response = await query_model_routed(chairman_config, messages, timeout=30.0)

    top_ranked_model = None
    top_ranked_avg = None
    if aggregate_rankings:
        top_ranked_model = aggregate_rankings[0].get("model")
        top_ranked_avg = aggregate_rankings[0].get("average_rank")

    if response is None:
        # Fallback if chairman fails: return the best available Stage 1 answer
        best_model = None
        if aggregate_rankings:
            best_model = aggregate_rankings[0].get("model")

        best_stage1 = None
        if best_model:
            for result in stage1_results:
                if result.get("model") == best_model:
                    best_stage1 = result
                    break

        if best_stage1 is None and stage1_results:
            best_stage1 = stage1_results[0]

        if best_stage1 is not None:
            return {
                "model": f"fallback/{best_stage1.get('model', 'unknown')}",
                "response": best_stage1.get("response", ""),
                "selected_model": best_stage1.get("model"),
                "selected_average_rank": top_ranked_avg,
            }

        return {
            "model": chairman_model_id,
            "response": "Error: Unable to generate final synthesis."
        }

    return {
        "model": chairman_model_id,
        "response": response.get('content', ''),
        "selected_model": top_ranked_model,
        "selected_average_rank": top_ranked_avg,
    }


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.

    Handles:
        - Markdown code fences wrapping the ranking (```...```)
        - Multiple numbering formats: "1. Response A", "1) Response A",
          "1 - Response A", "Response A - 1st", bare "Response A"
        - Conversational filler before/after the "FINAL RANKING:" section
        - Duplicate entries (deduplicates while preserving order)

    Args:
        ranking_text: The full text response from the model

    Returns:
        List of response labels in ranked order
    """

    def _unique_in_order(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def _strip_code_fences(text: str) -> str:
        """Remove markdown code fences (``` ... ```) from text."""
        return re.sub(r'```[a-zA-Z]*\n?', '', text).strip()

    def _extract_rankings_from_section(section: str) -> List[str]:
        """Try multiple formats to extract Response labels from a section."""
        section = _strip_code_fences(section)

        # Format 1: "1. Response A" (strict numbered list)
        matches = re.findall(r'\d+\.\s*(Response [A-Z])', section)
        if matches:
            return _unique_in_order(matches)

        # Format 2: "1) Response A"
        matches = re.findall(r'\d+\)\s*(Response [A-Z])', section)
        if matches:
            return _unique_in_order(matches)

        # Format 3: "1 - Response A"
        matches = re.findall(r'\d+\s*-\s*(Response [A-Z])', section)
        if matches:
            return _unique_in_order(matches)

        # Format 4: "Response A - 1st" / "Response A - 2nd" etc.
        matches = re.findall(r'(Response [A-Z])\s*-\s*\d+(?:st|nd|rd|th)?', section)
        if matches:
            return _unique_in_order(matches)

        # Format 5: Bare "Response X" on its own line or in text
        matches = re.findall(r'Response [A-Z]', section)
        if matches:
            return _unique_in_order(matches)

        return []

    # ── Primary: look for "FINAL RANKING:" section and IGNORE everything outside ──
    # Case-insensitive match for "FINAL RANKING:"
    ranking_match = re.search(r'FINAL\s+RANKING\s*:', ranking_text, re.IGNORECASE)
    if ranking_match:
        ranking_section = ranking_text[ranking_match.end():]
        result = _extract_rankings_from_section(ranking_section)
        if result:
            return result

    # ── Last resort: no FINAL RANKING section found ──
    # Strip code fences from entire text and try to find any Response labels
    cleaned = _strip_code_fences(ranking_text)
    matches = re.findall(r'Response [A-Z]', cleaned)
    return _unique_in_order(matches)


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts with model name and average rank, sorted best to worst
    """
    from collections import defaultdict

    # Track positions for each model
    model_positions = defaultdict(list)

    max_labels = len(label_to_model)

    for ranking in stage2_results:
        evaluator_model = ranking.get('model')
        parsed_ranking = ranking.get('parsed_ranking') or parse_ranking_from_text(ranking['ranking'])

        # Safety guard: de-duplicate and cap list length so malformed model
        # output cannot inflate vote counts.
        deduped: List[str] = []
        seen = set()
        for label in parsed_ranking:
            if label in seen:
                continue
            seen.add(label)
            deduped.append(label)
            if len(deduped) >= max_labels:
                break

        for position, label in enumerate(deduped, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                # Ignore a model's vote for itself to reduce self-recognition bias.
                if evaluator_model == model_name:
                    continue
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for a conversation based on the first user message.

    Args:
        user_query: The first user message

    Returns:
        A short title (3-5 words)
    """
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    # Use gemini-2.5-flash for title generation (fast and cheap) via legacy string API
    response = await query_model("google/gemini-2.5-flash", messages, timeout=30.0)

    if response is None:
        # Fallback to a generic title
        return "New Conversation"

    title = response.get('content', 'New Conversation').strip()

    # Clean up the title - remove quotes, limit length
    title = title.strip('"\'')

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title


async def run_full_council(user_query: str) -> Tuple[List, List, Dict, Dict]:
    """
    Run the complete 3-stage council process.

    Args:
        user_query: The user's question

    Returns:
        Tuple of (stage1_results, stage2_results, stage3_result, metadata)
    """
    # Stage 1: Collect individual responses
    stage1_results = await stage1_collect_responses(user_query)

    # If no models responded successfully, return error
    if not stage1_results:
        return [], [], {
            "model": "error",
            "response": "All models failed to respond. Please try again."
        }, {}

    # Stage 2: Collect rankings
    stage2_results, label_to_model = await stage2_collect_rankings(user_query, stage1_results)

    # Calculate aggregate rankings
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Stage 3: Synthesize final answer
    stage3_result = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results,
        aggregate_rankings,
    )

    # Prepare metadata
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings
    }

    return stage1_results, stage2_results, stage3_result, metadata
