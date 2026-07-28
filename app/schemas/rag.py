from pydantic import BaseModel, Field, field_validator


class RAGQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class RAGSource(BaseModel):
    text: str
    source: str
    score: float


class RAGQueryResponse(BaseModel):
    answer: str
    top_score: float
    sources: list[RAGSource]
