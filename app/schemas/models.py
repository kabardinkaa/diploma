from pydantic import BaseModel, ConfigDict, Field


class ModelInfo(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "openai/gpt-5.4-mini",
                "provider": "openrouter",
                "input_price_usd_per_1m_tokens": None,
                "output_price_usd_per_1m_tokens": None,
            }
        }
    )

    id: str = Field(
        ...,
        description="Server-configured model identifier",
        examples=["openai/gpt-5.4-mini"],
    )
    provider: str = Field(
        ...,
        description="Configured OpenAI-compatible provider",
        examples=["openrouter"],
    )
    input_price_usd_per_1m_tokens: float | None = Field(
        default=None,
        description="Цена входных токенов за 1M токенов",
    )
    output_price_usd_per_1m_tokens: float | None = Field(
        default=None,
        description="Цена выходных токенов за 1M токенов",
    )
