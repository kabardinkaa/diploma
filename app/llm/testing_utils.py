import json
import re
from typing import Any


JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
ALLOWED_ROLES = {"system", "user", "assistant"}


def escape_template_braces(text: str) -> str:
    """Escape braces so user text cannot break f-string/template-like prompts."""
    return text.replace("{", "{{").replace("}", "}}")


def build_chat_messages(
    system: str,
    user: str,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build chat messages in a stable role order: system → history → user."""
    messages: list[dict[str, str]] = []

    if system:
        messages.append({"role": "system", "content": system})

    for message in history or []:
        role = message.get("role", "")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"Unsupported role: {role}")

        messages.append(
            {
                "role": role,
                "content": escape_template_braces(message.get("content", "")),
            }
        )

    messages.append({"role": "user", "content": escape_template_braces(user)})
    return messages


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract JSON object from raw JSON text or markdown ```json fences."""
    candidate = text.strip()

    fence_match = JSON_FENCE_RE.search(candidate)
    if fence_match:
        candidate = fence_match.group(1).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("Malformed JSON response") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object")

    return parsed


def extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Normalize OpenAI-like tool_calls into plain dictionaries."""
    choices = getattr(response, "choices", None) or response.get("choices", [])
    if not choices:
        return []

    first_choice = choices[0]
    message = getattr(first_choice, "message", None) or first_choice.get("message", {})
    tool_calls = getattr(message, "tool_calls", None) or message.get("tool_calls") or []

    normalized: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        function = getattr(tool_call, "function", None) or tool_call.get("function", {})
        normalized.append(
            {
                "id": getattr(tool_call, "id", None) or tool_call.get("id"),
                "type": getattr(tool_call, "type", None) or tool_call.get("type"),
                "name": getattr(function, "name", None) or function.get("name"),
                "arguments": getattr(function, "arguments", None) or function.get("arguments"),
            }
        )

    return normalized


def calculate_llm_cost(
    usage: Any,
    input_price_per_1k: float,
    output_price_per_1k: float,
) -> float:
    """Calculate LLM cost from usage tokens and per-1k token prices."""
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0

    return round(
        (prompt_tokens / 1000 * input_price_per_1k)
        + (completion_tokens / 1000 * output_price_per_1k),
        6,
    )