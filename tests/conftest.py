from typing import Generator
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from backend import app
from backend.dependencies import chat_completion
from backend.settings import settings


@pytest.fixture(autouse=True)
def disable_background_provider_checks() -> Generator[None, None, None]:
    previous = settings.CHECK_WORKING_PROVIDERS
    settings.CHECK_WORKING_PROVIDERS = False
    try:
        yield
    finally:
        settings.CHECK_WORKING_PROVIDERS = previous


@pytest.fixture(scope="function")
def client() -> Generator[TestClient, None, None]:
    chat = Mock()
    chat.create.return_value = "response"
    app.dependency_overrides[chat_completion] = lambda: chat

    with TestClient(app) as client:
        yield client
