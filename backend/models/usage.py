from __future__ import annotations

from pydantic import BaseModel, Field


class PromptTokenDetails(BaseModel):
    """Prompt token usage breakdown (matches g4f.client.stubs.PromptTokenDetails)."""

    cached_tokens: int = Field(0)
    audio_tokens: int = Field(0)


class CompletionTokenDetails(BaseModel):
    """Completion token usage breakdown (matches g4f.client.stubs.CompletionTokenDetails)."""

    reasoning_tokens: int = Field(0)
    image_tokens: int = Field(0)
    audio_tokens: int = Field(0)


class Usage(BaseModel):
    """Token usage statistics from the provider (matches g4f.client.stubs.UsageModel)."""

    prompt_tokens: int = Field(0)
    completion_tokens: int = Field(0)
    total_tokens: int = Field(0)
    prompt_tokens_details: PromptTokenDetails | None = Field(None)
    completion_tokens_details: CompletionTokenDetails | None = Field(None)
