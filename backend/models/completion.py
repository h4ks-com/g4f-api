from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from backend.models.usage import Usage


class Message(BaseModel):
    role: Literal["user", "assistant"] = Field(
        ..., description="Who is sending the message"
    )
    content: str = Field(..., description="Content of the message")


class ToolFunction(BaseModel):
    """Function definition for tool calling."""

    name: str = Field(..., description="The name of the function to be called.")
    description: str = Field("", description="A description of what the function does.")
    # JSON Schema is an open-ended recursive structure (type, properties, items,
    # $ref, allOf, etc.) that can't be meaningfully constrained to a fixed
    # Pydantic model without losing generality.
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="The parameters the function accepts as JSON Schema.",
    )


class ToolCallFunction(BaseModel):
    """The function call details from a model response."""

    name: str = Field(..., description="The name of the function to call.")
    arguments: str = Field(
        ..., description="The arguments to call the function with, as a JSON string."
    )


class ToolDefinition(BaseModel):
    """A tool/function definition the model can call."""

    type: Literal["function"] = Field("function", description="The type of the tool.")
    function: ToolFunction = Field(..., description="The function to call.")


class ToolCall(BaseModel):
    """A tool call returned by the model."""

    id: str = Field(..., description="The ID of the tool call.")
    type: Literal["function"] = Field(
        "function", description="The type of the tool call."
    )
    function: ToolCallFunction = Field(..., description="The function that was called.")


class ToolChoiceFunction(BaseModel):
    """Specifies a particular function for tool_choice."""

    name: str = Field(..., description="The name of the function.")


class ToolChoice(BaseModel):
    """Specifies which tool the model should use."""

    type: Literal["function"] = Field(
        "function", description="The type of the tool choice."
    )
    function: ToolChoiceFunction = Field(
        ..., description="The specific function to call."
    )


class CompletionRequest(BaseModel):
    messages: list[Message] = Field(
        ..., description="List of messages to use for completion"
    )
    tools: list[ToolDefinition] | None = Field(
        None,
        description="Optional list of tool definitions for function calling.",
    )
    tool_choice: Literal["auto", "none", "required"] | ToolChoice | None = Field(
        None,
        description='Controls tool use: "auto", "none", "required", or a specific tool.',
    )
    model_config = ConfigDict(extra="ignore")


class CompletionModel(BaseModel):
    name: str
    supported_provider_names: set[str]

    @field_serializer("supported_provider_names")
    @classmethod
    def serialize_supported_provider_names(cls, v: set[str]) -> list[str]:
        return list(v)


class CompletionProvider(BaseModel):
    name: str
    url: str
    supported_models: set[str]
    supports_tools: bool = Field(
        False,
        description="Whether this provider supports tool/function calling.",
    )

    @field_serializer("supported_models")
    @classmethod
    def serialize_supported_models(cls, v: set[str]) -> list[str]:
        return list(v)


class ProviderFailure(BaseModel):
    error_type: str
    error_message: str
    traceback: str
    timestamp: datetime
    model_used: str
    messages: list[dict[str, str]]
    response: dict[str, Any] | None = None

    @field_validator("response", mode="before")
    @classmethod
    def coerce_response(cls, v: Any) -> dict[str, Any] | None:
        if v is None or isinstance(v, dict):
            return v
        try:
            return {
                "status": getattr(v, "status", None),
                "url": str(getattr(v, "url", "")),
            }
        except Exception:
            return {"raw": str(type(v))}


class ProviderFailuresResponse(BaseModel):
    failures: dict[str, ProviderFailure]
    total_failed_providers: int
    description: str


class CompletionResponse(BaseModel):
    completion: str = Field(..., description="Completion of the messages")
    provider: str | None = Field(None, description="Provider used for completion")
    model: str | None = Field(None, description="Model used for completion")
    tool_calls: list[ToolCall] | None = Field(
        None, description="Tool calls made by the model when tools are requested."
    )
    finish_reason: str | None = Field(
        None, description="Reason for completion (stop, tool_calls, etc.)."
    )
    id: str | None = Field(None, description="Completion ID (OpenAI-compatible).")
    usage: "Usage | None" = Field(None, description="Token usage statistics.")
