from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: int | None = None


class ChatResponse(BaseModel):
    conversation_id: int
    answer: str
    context: list[str] = Field(default_factory=list)
    pipeline: str = "rag"
    route_reason: str = ""
    query_expansions: list[str] = Field(default_factory=list)
    context_coverage: float | None = None
    context_sufficient: bool | None = None
