from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    id: str = Field(..., description="ID модели")
    provider: str = Field(..., description="Провайдер модели")
    input_price_usd_per_1m_tokens: float | None = Field(
        default=None,
        description="Цена входных токенов за 1M токенов",
    )
    output_price_usd_per_1m_tokens: float | None = Field(
        default=None,
        description="Цена выходных токенов за 1M токенов",
    )


AVAILABLE_MODELS = [
    ModelInfo(
        id="gpt-4.1-mini",
        provider="openai",
        input_price_usd_per_1m_tokens=0.4,
        output_price_usd_per_1m_tokens=1.6,
    ),
    ModelInfo(
        id="gpt-4o-mini",
        provider="openai",
        input_price_usd_per_1m_tokens=0.15,
        output_price_usd_per_1m_tokens=0.6,
    ),
    ModelInfo(
        id="openrouter/free",
        provider="openrouter",
        input_price_usd_per_1m_tokens=0,
        output_price_usd_per_1m_tokens=0,
    ),
]