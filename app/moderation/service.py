import hashlib
import re
from pathlib import Path
from typing import Any, Literal

import structlog
import yaml
from pydantic import BaseModel, Field

from app.observability.pii import redact_pii


logger = structlog.get_logger("llm-service")


class ModerationResult(BaseModel):
    allowed: bool = True
    categories: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    blocked_by: str = "none"


class ModerationRule(BaseModel):
    category: str
    pattern: str
    reason: str


class ModerationService:
    def __init__(
        self,
        keywords_path: Path | None = None,
        openai_enabled: bool = False,
        openai_client: Any | None = None,
    ) -> None:
        self.keywords_path = keywords_path or Path(__file__).with_name(
            "moderation_keywords.yaml"
        )
        self.openai_enabled = openai_enabled
        self.openai_client = openai_client
        self._rules = self._load_rules()

    def _load_rules(self) -> dict[str, list[ModerationRule]]:
        data = yaml.safe_load(
            self.keywords_path.read_text(encoding="utf-8")
        ) or {}

        return {
            section: [
                ModerationRule.model_validate(item)
                for item in (data.get(section, {}).get("banned") or [])
            ]
            for section in ("input", "output")
        }

    async def check_input(self, content: str) -> ModerationResult:
        return await self._check(content, "input")

    async def check_output(self, content: str) -> ModerationResult:
        return await self._check(content, "output")

    async def _check(
        self,
        content: str,
        direction: Literal["input", "output"],
    ) -> ModerationResult:
        result = self._check_local(content, direction)

        if not result.allowed:
            self.log_incident(content, result, direction)
            return result

        # Optional hook: if a configured OpenAI client is passed in later,
        # local moderation remains the no-network default for the homework.
        if self.openai_enabled and self.openai_client is not None:
            return result

        return result

    def _check_local(
        self,
        content: str,
        direction: Literal["input", "output"],
    ) -> ModerationResult:
        categories: list[str] = []
        reasons: list[str] = []

        for rule in self._rules.get(direction, []):
            if re.search(rule.pattern, content, flags=re.IGNORECASE):
                categories.append(rule.category)
                reasons.append(rule.reason)

        if categories:
            return ModerationResult(
                allowed=False,
                categories=sorted(set(categories)),
                reasons=sorted(set(reasons)),
                blocked_by="local_keyword_regex",
            )

        return ModerationResult()

    def log_incident(
        self,
        content: str,
        result: ModerationResult,
        direction: str,
    ) -> None:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

        logger.warning(
            "moderation.blocked",
            direction=direction,
            text_hash=digest,
            masked_preview=redact_pii(content)[:160],
            categories=result.categories,
            reasons=result.reasons,
            blocked_by=result.blocked_by,
        )
