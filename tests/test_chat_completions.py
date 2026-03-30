from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from backend import app
from backend.dependencies import chat_completion

COMPLETIONS_PATH = "/api/completions"


@pytest.fixture
def client():
    """TestClient with no overrides — for testing real behavior like provider endpoints."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_chat_client():
    """TestClient with mocked g4f.ChatCompletion."""
    chat = Mock()
    chat.create.return_value = "mocked response"
    app.dependency_overrides[chat_completion] = lambda: chat
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestBackwardsCompatibility:
    """Existing plain text behavior must be unchanged."""

    def test_plain_completion(self, mock_chat_client):
        r = mock_chat_client.post(
            COMPLETIONS_PATH, json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert r.status_code == 200
        data = r.json()
        assert data["completion"] == "mocked response"

    def test_nofail_auto_select(self, mock_chat_client):
        r = mock_chat_client.post(
            COMPLETIONS_PATH, json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert r.status_code == 200
        assert "completion" in r.json()

    def test_extra_params_ignored(self, mock_chat_client):
        r = mock_chat_client.post(
            COMPLETIONS_PATH,
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "unexpected": "ignored",
            },
        )
        assert r.status_code == 200


def _mock_g4f_response(content=None, tool_calls=None, finish_reason="stop"):
    """Build a mock g4f ChatCompletion response."""
    mock_msg = MagicMock()
    mock_msg.content = content
    mock_msg.tool_calls = tool_calls

    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_choice.finish_reason = finish_reason

    mock_prompt_details = MagicMock()
    mock_prompt_details.cached_tokens = 0
    mock_prompt_details.audio_tokens = 0

    mock_completion_details = MagicMock()
    mock_completion_details.reasoning_tokens = 0
    mock_completion_details.image_tokens = 0
    mock_completion_details.audio_tokens = 0

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_usage.total_tokens = 15
    mock_usage.prompt_tokens_details = mock_prompt_details
    mock_usage.completion_tokens_details = mock_completion_details

    mock_response = MagicMock()
    mock_response.id = "chatcmpl-test123"
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    return mock_response


def _mock_async_client(mock_response):
    """Return a mock AsyncClient whose create returns the given response."""
    instance = MagicMock()
    instance.chat.completions.create = AsyncMock(return_value=mock_response)
    return instance


def _mock_nofail(model="openai-fast", provider="PollinationsAI"):
    """Return a mock NofailParams object."""
    return MagicMock(model=model, provider=provider)


_MOCK_PROVIDER_CLASS = MagicMock(__name__="PollinationsAI")

# Decorators common to all tool-calling tests that hit the AsyncClient path.
_TOOL_PATCHES = [
    patch("backend.routes.get_nofail_params", return_value=_mock_nofail()),
    patch("backend.routes._get_provider_class", return_value=_MOCK_PROVIDER_CLASS),
    patch("backend.routes.AsyncClient"),
]


def _apply_tool_patches(func):
    """Decorator that applies all tool-related patches in the correct order."""
    for p in reversed(_TOOL_PATCHES):
        func = p(func)
    return func


class TestToolCalling:
    """Tool calling uses AsyncClient and routes through the same retry loop."""

    @_apply_tool_patches
    def test_tools_passed_to_async_client(
        self, mock_cls, mock_get_provider, mock_nofail, client
    ):
        """Verify that when tools are in the request, AsyncClient.create is called
        with the tools forwarded."""
        mock_cls.return_value = _mock_async_client(_mock_g4f_response(content="ok"))

        r = client.post(
            COMPLETIONS_PATH,
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "f", "parameters": {"type": "object"}},
                    }
                ],
            },
        )

        assert r.status_code == 200
        call_kwargs = mock_cls.return_value.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1

    @_apply_tool_patches
    def test_tool_choice_forwarded(
        self, mock_cls, mock_get_provider, mock_nofail, client
    ):
        """Verify tool_choice string is forwarded to the g4f client."""
        mock_cls.return_value = _mock_async_client(_mock_g4f_response(content="ok"))

        client.post(
            COMPLETIONS_PATH,
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "f", "parameters": {"type": "object"}},
                    }
                ],
                "tool_choice": "auto",
            },
        )

        call_kwargs = mock_cls.return_value.chat.completions.create.call_args[1]
        assert call_kwargs["tool_choice"] == "auto"

    @_apply_tool_patches
    def test_specific_tool_choice_forwarded(
        self, mock_cls, mock_get_provider, mock_nofail, client
    ):
        """Verify a specific tool_choice dict is forwarded correctly."""
        mock_cls.return_value = _mock_async_client(_mock_g4f_response(content="ok"))

        client.post(
            COMPLETIONS_PATH,
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "f", "parameters": {"type": "object"}},
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "f"}},
            },
        )

        call_kwargs = mock_cls.return_value.chat.completions.create.call_args[1]
        assert call_kwargs["tool_choice"]["type"] == "function"
        assert call_kwargs["tool_choice"]["function"]["name"] == "f"

    @_apply_tool_patches
    def test_nofail_filters_to_tool_capable(
        self, mock_cls, mock_get_provider, mock_nofail, client
    ):
        """When tools are requested, get_nofail_params should be called with
        require_tools=True."""
        mock_cls.return_value = _mock_async_client(_mock_g4f_response(content="ok"))

        client.post(
            COMPLETIONS_PATH,
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "f", "parameters": {"type": "object"}},
                    }
                ],
            },
        )

        mock_nofail.assert_called_once_with(require_tools=True)

    @_apply_tool_patches
    def test_response_has_tool_calls_structure(
        self, mock_cls, mock_get_provider, mock_nofail, client
    ):
        """When g4f returns tool_calls, the response has the right structure."""
        mock_tc = MagicMock()
        mock_tc.id = "call_abc"
        mock_tc.type = "function"
        mock_tc.function.name = "get_weather"
        mock_tc.function.arguments = '{"location": "Paris"}'

        mock_cls.return_value = _mock_async_client(
            _mock_g4f_response(tool_calls=[mock_tc], finish_reason="tool_calls")
        )

        r = client.post(
            COMPLETIONS_PATH,
            json={
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "f", "parameters": {"type": "object"}},
                    }
                ],
            },
        )

        assert r.status_code == 200
        data = r.json()
        assert data["finish_reason"] == "tool_calls"
        tc = data["tool_calls"][0]
        assert tc["id"] == "call_abc"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"location": "Paris"}'


class TestSupportsToolsField:
    def test_providers_have_supports_tools(self, client):
        r = client.get("/api/providers")
        assert r.status_code == 200
        data = r.json()
        for name, provider in data.items():
            assert "supports_tools" in provider, f"{name} missing supports_tools"


class TestPydanticModels:
    def test_request_without_tools(self):
        from backend.models import CompletionRequest, Message

        req = CompletionRequest(messages=[Message(role="user", content="hi")])
        assert req.tools is None
        assert req.tool_choice is None

    def test_request_with_tools(self):
        from backend.models import CompletionRequest, Message, ToolDefinition

        req = CompletionRequest(
            messages=[Message(role="user", content="hi")],
            tools=[
                ToolDefinition(
                    type="function",
                    function={"name": "f", "parameters": {"type": "object"}},
                )
            ],
            tool_choice="auto",
        )
        assert len(req.tools) == 1
        assert req.tools[0].function.name == "f"

    def test_response_new_fields_nullable(self):
        from backend.models import CompletionResponse

        resp = CompletionResponse(completion="hi", model="gpt-4", provider="Test")
        assert resp.tool_calls is None
        assert resp.finish_reason is None
        assert resp.id is None
        assert resp.usage is None
