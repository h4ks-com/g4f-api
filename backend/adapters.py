# Description: This file contains the adapter functions that are used to convert data from one format to another.

from collections.abc import Callable
from urllib.parse import unquote


def url_decode(text: str) -> str:
    """
    Decodes a URL-encoded string.

    Args:
        text (str): A URL-encoded string.

    Returns:
        str: A decoded string.
    """
    return unquote(text)


ADAPTERS_MAP: dict[str, Callable[[str], str]] = {
    "Ai4Chat": url_decode,
}


def extract_openai_content(response: dict | str) -> str:
    """Extract content from OpenAI-style response structure."""
    if isinstance(response, str):
        return response

    if isinstance(response, dict):
        # Handle OpenAI-style response: choices[0].message.content
        if "choices" in response and len(response["choices"]) > 0:
            message = response["choices"][0].get("message", {})
            if "content" in message:
                return message["content"]

        # If structure doesn't match, fallback to str conversion
        return str(response)

    return str(response)


def adapt_response(model_name: str, response: dict | str) -> str:
    """
    Adapts the response based on the model, handling both string and dict responses.

    Args:
        model_name: The name of the model.
        response: The response to adapt (string or dict with OpenAI structure).

    Returns:
        The adapted response text.
    """
    # First extract content from structured responses
    text = extract_openai_content(response)

    # Then apply model-specific adapters
    adapter = ADAPTERS_MAP.get(model_name)
    if adapter is None:
        return text
    return adapter(text)
