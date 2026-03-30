from __future__ import annotations

from pydantic import BaseModel, Field


class UsageDetails(BaseModel):
    """Detailed token usage breakdown."""

    cached_tokens: int | None = Field(None)
    audio_tokens: int | None = Field(None)
    reasoning_tokens: int | None = Field(None)


class Usage(BaseModel):
    """Token usage statistics from the provider."""

    prompt_tokens: int = Field(0)
    completion_tokens: int = Field(0)
    total_tokens: int = Field(0)
    prompt_tokens_details: UsageDetails | None = Field(None)
    completion_tokens_details: UsageDetails | None = Field(None)
