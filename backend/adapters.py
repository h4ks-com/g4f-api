from collections.abc import Callable
from urllib.parse import unquote

import yaml


def url_decode(text: str) -> str:
    return unquote(text)


ADAPTERS_MAP: dict[str, Callable[[str], str]] = {
    "Ai4Chat": url_decode,
}


def _extract_content_from_payload(payload: dict) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None

    message = first_choice.get("message")
    if not isinstance(message, dict) or "content" not in message:
        return None

    content = message.get("content")
    if content is None:
        return ""
    return str(content)


def _iter_mapping_candidates(text: str):
    start_index: int | None = None
    depth = 0
    quote_char: str | None = None
    escaped = False

    for index, char in enumerate(text):
        if quote_char is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote_char:
                quote_char = None
            continue

        if char in {'"', "'"}:
            quote_char = char
            continue

        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
            continue

        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_index is not None:
                yield text[start_index : index + 1]
                start_index = None


def extract_openai_content(response: dict | str) -> str:
    if isinstance(response, dict):
        content = _extract_content_from_payload(response)
        if content is not None:
            return content
        return str(response)

    if not isinstance(response, str):
        return str(response)

    # Some providers concatenate dicts as strings: str(request) + str(response) + content
    if "'choices'" not in response and '"choices"' not in response:
        return response

    for candidate in _iter_mapping_candidates(response):
        if "choices" not in candidate:
            continue
        try:
            parsed = yaml.safe_load(candidate)
        except yaml.YAMLError:
            continue
        if not isinstance(parsed, dict):
            continue

        content = _extract_content_from_payload(parsed)
        if content is not None:
            return content

    return response


def adapt_response(model_name: str, response: dict | str) -> str:
    text = extract_openai_content(response)
    adapter = ADAPTERS_MAP.get(model_name)
    if adapter is None:
        return text
    return adapter(text)
