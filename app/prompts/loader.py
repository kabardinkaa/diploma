from functools import lru_cache
from pathlib import Path

from jinja2 import Template


PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=8)
def render_system_prompt(version: str = "v1", **context) -> str:
    """Render system prompt from app/prompts/system_v*.j2."""
    prompt_path = PROMPTS_DIR / f"system_{version}.j2"
    text = prompt_path.read_text(encoding="utf-8")
    return Template(text).render(**context)


@lru_cache(maxsize=16)
def load_tool_description(tool_name: str) -> str:
    """Load long tool description from app/prompts/tools/<tool_name>.md."""
    description_path = PROMPTS_DIR / "tools" / f"{tool_name}.md"
    return description_path.read_text(encoding="utf-8").strip()
