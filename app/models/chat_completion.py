from time import time
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field


class ChatMessage(BaseModel):
    content: str | None
    role: Literal["system", "user", "assistant"]


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str
    stream: bool = False
    temperature: float = 1
    top_p: float = 1


class ChatChoiceMessage(BaseModel):
    finish_reason: Literal["stop"]
    index: int
    message: ChatMessage


class ChatChoiceChunk(BaseModel):
    delta: ChatMessage
    finish_reason: Literal["stop", None] = None
    index: int


class Usage(BaseModel):
    completion_tokens: int = 0
    prompt_tokens: int = 0

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.completion_tokens + self.prompt_tokens


class ChatCompletionResponse(BaseModel):
    choices: list[ChatChoiceMessage | ChatChoiceChunk]
    created: int = Field(default_factory=lambda: int(time()))
    id: str = Field(default_factory=lambda: f"deepsearch-4-all-{uuid4().hex}")
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    usage: Usage | None
