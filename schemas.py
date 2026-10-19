
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="User's natural language question"
    )


class SourceInfo(BaseModel):
    source: str
    page: int | str


class ChatResponse(BaseModel):
    answer: str
    stand_ids: list[int] = Field(default_factory=list)
    feature_ids: list[int] = Field(default_factory=list)
    sources: list[SourceInfo] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
