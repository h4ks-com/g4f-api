from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from backend import app
from backend.adapters import extract_openai_content
from backend.dependencies import chat_completion, provider_and_models


def test_extract_openai_content_from_concatenated_payloads() -> None:
    """Recover assistant content from concatenated provider payload fragments."""
    response = (
        "{'request': {'messages': []}}"
        "{'choices': [{'message': {'content': 'hello from provider'}}]}"
        "trailing text"
    )

    assert extract_openai_content(response) == "hello from provider"


@pytest.mark.parametrize("base_path", ["/api/", "/v1/"])
def test_openai_sdk_chat_completions_uses_compat_route(base_path: str) -> None:
    """Verify both compatibility routes work with the OpenAI Python SDK."""
    chat = Mock()
    chat.create.return_value = "sdk response"
    app.dependency_overrides[chat_completion] = lambda: chat
    model = provider_and_models.all_model_names[0]

    try:
        with TestClient(app) as client:
            sdk_client = OpenAI(
                api_key="test-key",
                base_url=str(client.base_url).rstrip("/") + base_path,
                http_client=client,
            )

            response = sdk_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert response.object == "chat.completion"
            assert response.model == model
            assert response.choices[0].message.content == "sdk response"
            assert response.choices[0].finish_reason == "stop"
    finally:
        app.dependency_overrides.pop(chat_completion, None)
