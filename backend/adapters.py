import ast
from collections.abc import Callable
from urllib.parse import unquote


def url_decode(text: str) -> str:
    return unquote(text)


ADAPTERS_MAP: dict[str, Callable[[str], str]] = {
    "Ai4Chat": url_decode,
}


def extract_openai_content(response: dict | str) -> str:
    if isinstance(response, dict):
        if "choices" in response and len(response["choices"]) > 0:
            message = response["choices"][0].get("message", {})
            if "content" in message:
                return message["content"]
        return str(response)

    if not isinstance(response, str):
        return str(response)

    # Some providers concatenate dicts as strings: str(request) + str(response) + content
    if "'choices'" not in response and '"choices"' not in response:
        return response

    parts = response.split("}{")

    for i, part in enumerate(parts):
        if i > 0:
            part = "{" + part
        if i < len(parts) - 1:
            part = part + "}"

        brace_count = 0
        dict_end = -1
        for j, char in enumerate(part):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    dict_end = j + 1
                    break

        if dict_end > 0:
            dict_str = part[:dict_end]
            try:
                parsed = ast.literal_eval(dict_str)
                if isinstance(parsed, dict) and "choices" in parsed:
                    if len(parsed["choices"]) > 0:
                        message = parsed["choices"][0].get("message", {})
                        if "content" in message:
                            return message["content"]
            except (ValueError, SyntaxError):
                continue

    return response


def adapt_response(model_name: str, response: dict | str) -> str:
    text = extract_openai_content(response)
    adapter = ADAPTERS_MAP.get(model_name)
    if adapter is None:
        return text
    return adapter(text)
