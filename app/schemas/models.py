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
