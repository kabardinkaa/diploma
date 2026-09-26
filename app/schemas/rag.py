from pydantic import BaseModel, ConfigDict, Field, field_validator


class RAGQueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"question": "Как подключиться к корпоративному VPN?"}
        }
    )

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question answered from the `corporate_rag` knowledge base",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class RAGSource(BaseModel):
    id: int = Field(ge=1, description="Citation number used in the answer")
    file_name: str = Field(description="Source document file name")
    page: int | None = None
    score: float
    snippet: str


class RAGQueryResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "Откройте VPN-клиент и войдите с корпоративной учётной записью [1].",
                "top_score": 0.874,
                "confident": True,
                "sources": [
                    {
                        "id": 1,
                        "file_name": "vpn_access.md",
                        "page": None,
                        "score": 0.874,
                        "snippet": "Инструкция по подключению к корпоративному VPN...",
                    }
                ],
            }
        }
    )

    answer: str
    top_score: float
    confident: bool
    sources: list[RAGSource]
