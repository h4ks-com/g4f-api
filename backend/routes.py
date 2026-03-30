import asyncio
import logging
from dataclasses import dataclass

import g4f
import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from g4f.client import AsyncClient
from g4f.client.stubs import ChatCompletion as G4fChatCompletion
from g4f.client.stubs import UsageModel as G4fUsageModel
from g4f.errors import ModelNotFoundError, ProviderNotWorkingError

from backend.adapters import adapt_response
from backend.background import (
    add_successful_provider,
    get_cached_successful_providers,
    provider_failures,
)
from backend.dependencies import (
    BEST_MODELS_ORDERED,
    CompletionParams,
    Message,
    UiCompletionRequest,
    chat_completion,
    provider_and_models,
)
from backend.errors import CustomValidationError
from backend.models import (
    CompletionRequest,
    CompletionResponse,
    CompletionTokenDetails,
    PromptTokenDetails,
    ProviderFailuresResponse,
    ToolCall,
    ToolCallFunction,
    Usage,
)
from backend.settings import TEMPLATES_PATH

logger = logging.getLogger(__name__)

router_root = APIRouter()
router_api = APIRouter(prefix="/api")
router_ui = APIRouter(prefix="/app")


def add_routers(app: FastAPI) -> None:
    app.include_router(router_root)
    app.include_router(router_api)
    app.include_router(router_ui)


@router_root.get("/")
def get_root():
    return RedirectResponse(url=router_ui.prefix)


@dataclass(frozen=True)
class NofailParams:
    model: str
    provider: str


def _tool_capable_providers() -> list[str]:
    """Return names of working providers that support tool calling."""
    return [
        p
        for p in provider_and_models.all_working_provider_names
        if provider_and_models.all_working_providers_map[p].supports_tools
    ]


def _is_provider_model_available(provider_name: str, model: str) -> bool:
    return (
        provider_name in provider_and_models.all_working_provider_names
        and model
        in provider_and_models.all_working_providers_map[provider_name].supported_models
    )


def _resolve_nofail_params(
    failed_combinations: set[tuple[str, str]] | None = None,
    offset: int = 0,
    require_tools: bool = False,
) -> NofailParams:
    """Shared nofail provider/model selection.

    When *failed_combinations* is provided, skip those (model, provider) pairs.
    """
    providers_to_check = list(provider_and_models.all_working_provider_names)
    if require_tools:
        providers_to_check = _tool_capable_providers()
        if not providers_to_check:
            raise HTTPException(
                status_code=422,
                detail="No tool-capable providers currently working.",
            )

    def _is_allowed(model: str, provider_name: str) -> bool:
        if failed_combinations is None:
            return True
        return (model, provider_name) not in failed_combinations

    for model in BEST_MODELS_ORDERED:
        try:
            default_provider = g4f.get_model_and_provider(model, None, False)[1]
        except (ModelNotFoundError, ProviderNotWorkingError):
            logger.warning("Model not found or not working: %s", model)
            continue

        if offset > 0:
            offset -= 1
            continue

        # Priority 1: model-filtered cached successes
        for provider_name in get_cached_successful_providers(model_filter=model):
            if (
                provider_name in providers_to_check
                and _is_provider_model_available(provider_name, model)
                and _is_allowed(model, provider_name)
            ):
                return NofailParams(model=model, provider=provider_name)

        # Priority 2: default provider
        if default_provider.__name__ in providers_to_check and _is_allowed(
            model, default_provider.__name__
        ):
            return NofailParams(model=model, provider=default_provider.__name__)

        # Priority 3: all cached successes (no model filter)
        for provider_name in get_cached_successful_providers():
            if (
                provider_name in providers_to_check
                and _is_provider_model_available(provider_name, model)
                and _is_allowed(model, provider_name)
            ):
                return NofailParams(model=model, provider=provider_name)

        # Priority 4: any working provider that has the model
        for provider_name in providers_to_check:
            if _is_provider_model_available(provider_name, model) and _is_allowed(
                model, provider_name
            ):
                return NofailParams(model=model, provider=provider_name)

    raise HTTPException(
        status_code=500, detail="Failed to find a model and provider to use"
    )


def get_nofail_params(offset: int = 0, require_tools: bool = False) -> NofailParams:
    return _resolve_nofail_params(
        failed_combinations=None, offset=offset, require_tools=require_tools
    )


def get_nofail_params_excluding_failed(
    failed_combinations: set[tuple[str, str]],
    offset: int = 0,
    require_tools: bool = False,
) -> NofailParams:
    return _resolve_nofail_params(
        failed_combinations=failed_combinations,
        offset=offset,
        require_tools=require_tools,
    )


def get_best_model_for_provider(provider_name: str) -> str:
    provider = provider_and_models.all_working_providers_map.get(provider_name)
    if provider is None:
        raise HTTPException(
            status_code=422, detail=f"Provider not found: {provider_name}"
        )
    models = list(provider.supported_models)
    if not models:
        raise HTTPException(
            status_code=422,
            detail=f"No models supported by provider: {provider_name}. Please specify a model.",
        )

    def _sort_key(model: str) -> int:
        return BEST_MODELS_ORDERED.index(model) if model in BEST_MODELS_ORDERED else 999

    models.sort(key=_sort_key)
    return models[0]


_public_ip_cache: str | None = None


async def get_public_ip() -> str | None:
    """Fetch public IP (cached for the process lifetime)."""
    global _public_ip_cache
    if _public_ip_cache is not None:
        return _public_ip_cache
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.ipify.org?format=json")
            response.raise_for_status()
            _public_ip_cache = response.json().get("ip")
            return _public_ip_cache
    except httpx.HTTPError:
        return None


def _get_provider_class(provider_name: str) -> type | None:
    for provider in g4f.Provider.__providers__:
        if provider.__name__ == provider_name:
            return provider
    return None


def _to_tool_calls(raw_tool_calls: list) -> list[ToolCall] | None:
    """Convert g4f ToolCallModel list to our typed ToolCall list."""
    if not raw_tool_calls:
        return None
    result = [
        ToolCall(
            id=tc.id,
            type=tc.type or "function",
            function=ToolCallFunction(
                name=tc.function.name,
                arguments=tc.function.arguments,
            ),
        )
        for tc in raw_tool_calls
    ]
    return result if result else None


def _to_usage(g4f_usage: G4fUsageModel) -> Usage:
    """Convert g4f UsageModel to our Usage model."""
    prompt_details = g4f_usage.prompt_tokens_details
    completion_details = g4f_usage.completion_tokens_details
    return Usage(
        prompt_tokens=g4f_usage.prompt_tokens,
        completion_tokens=g4f_usage.completion_tokens,
        total_tokens=g4f_usage.total_tokens,
        prompt_tokens_details=PromptTokenDetails(
            cached_tokens=prompt_details.cached_tokens,
            audio_tokens=getattr(prompt_details, "audio_tokens", 0),
        ),
        completion_tokens_details=CompletionTokenDetails(
            reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0),
            image_tokens=getattr(completion_details, "image_tokens", None),
            audio_tokens=getattr(completion_details, "audio_tokens", None),
        ),
    )


async def _call_with_tools(
    completion: CompletionRequest, model_name: str, provider_name: str
) -> CompletionResponse:
    """Call g4f via AsyncClient for tool calling support. Fully async."""
    provider_class = _get_provider_class(provider_name)
    if provider_class is None:
        raise HTTPException(
            status_code=500, detail=f"Provider not found: {provider_name}"
        )

    client = AsyncClient()
    kwargs = {"stream": False}
    if completion.tools:
        kwargs["tools"] = [t.model_dump() for t in completion.tools]
    if completion.tool_choice is not None:
        if isinstance(completion.tool_choice, str):
            kwargs["tool_choice"] = completion.tool_choice
        else:
            kwargs["tool_choice"] = completion.tool_choice.model_dump()

    response: G4fChatCompletion = await client.chat.completions.create(
        model=model_name,
        provider=provider_class,
        messages=[msg.model_dump() for msg in completion.messages],
        **kwargs,
    )

    choice = response.choices[0]
    msg = choice.message

    return CompletionResponse(
        completion=msg.content or "",
        model=model_name,
        provider=provider_name,
        tool_calls=_to_tool_calls(msg.tool_calls),
        finish_reason=choice.finish_reason,
        id=response.id,
        usage=_to_usage(response.usage),
    )


async def _call_plain(
    completion: CompletionRequest,
    model_name: str,
    provider_name: str,
    chat: type[g4f.ChatCompletion],
) -> CompletionResponse:
    """Call g4f via ChatCompletion.create for plain text responses (non-blocking)."""
    response = await asyncio.to_thread(
        chat.create,
        model=model_name,
        provider=provider_name,
        messages=[msg.model_dump() for msg in completion.messages],
        stream=False,
    )
    if not isinstance(response, str | dict):
        raise CustomValidationError(
            "Unexpected response type from g4f.ChatCompletion.create",
            error={"response": str(response)},
        )

    adapted_text = adapt_response(model_name, response)
    return CompletionResponse(
        completion=adapted_text,
        model=model_name,
        provider=provider_name,
    )


@router_api.post("/completions")
async def post_completion(
    completion: CompletionRequest,
    params: CompletionParams = Depends(),
    chat: type[g4f.ChatCompletion] = Depends(chat_completion),
) -> CompletionResponse:
    has_tools = completion.tools is not None and len(completion.tools) > 0

    nofail = False
    if params.model is None:
        if params.provider is None:
            nofail_params = get_nofail_params(require_tools=has_tools)
            model_name, provider_name = nofail_params.model, nofail_params.provider
            nofail = True
        else:
            provider_name = params.provider
            model_name = get_best_model_for_provider(provider_name)
    else:
        model_name = params.model
        provider_name = params.provider

    ip_detected_response: CompletionResponse | None = None
    failed_combinations: set[tuple[str, str]] = set()

    for attempt in range(10):
        logger.debug(
            "Trying model=%s provider=%s (attempt %d)",
            model_name,
            provider_name,
            attempt,
        )
        try:
            if has_tools:
                completion_response = await _call_with_tools(
                    completion, model_name, provider_name
                )
            else:
                completion_response = await _call_plain(
                    completion, model_name, provider_name, chat
                )

            # Handle empty responses in nofail mode
            if (
                (
                    not completion_response.completion
                    or completion_response.completion.strip() == ""
                )
                and not completion_response.tool_calls
                and nofail
            ):
                failed_combinations.add((model_name, provider_name))
                nofail_params = get_nofail_params_excluding_failed(
                    failed_combinations, attempt, require_tools=has_tools
                )
                model_name, provider_name = nofail_params.model, nofail_params.provider
                continue

            # HACK: Workaround for IP ban from some providers
            if not has_tools:
                ip = await get_public_ip()
                if ip is not None and ip in completion_response.completion.lower():
                    if ip_detected_response is None:
                        ip_detected_response = completion_response
                    continue

            add_successful_provider(provider_name, model_name)
            return completion_response

        except CustomValidationError:
            raise
        except Exception as e:
            if not nofail:
                raise e
            logger.warning(
                "Provider %s/%s failed (attempt %d): %s",
                model_name,
                provider_name,
                attempt,
                e,
            )
            failed_combinations.add((model_name, provider_name))
            nofail_params = get_nofail_params_excluding_failed(
                failed_combinations, attempt, require_tools=has_tools
            )
            model_name, provider_name = nofail_params.model, nofail_params.provider

    if ip_detected_response is not None:
        add_successful_provider(
            ip_detected_response.provider, ip_detected_response.model
        )
        return ip_detected_response

    raise HTTPException(
        status_code=500,
        detail=f"Failed to get a response from the provider. Last tried model: {model_name} and provider: {provider_name}",
    )


@router_api.get("/providers")
def get_list_providers():
    return provider_and_models.all_working_providers_map


@router_api.get("/models")
def get_list_models():
    return provider_and_models.all_models_map


@router_api.get("/health")
def get_health_check():
    return {"status": "ok"}


@router_api.get("/provider-failures")
def get_provider_failures() -> ProviderFailuresResponse:
    """
    Returns detailed failure information for providers that have failed during the last test cycle.
    Includes error messages, backtraces, and response details.
    """
    return ProviderFailuresResponse(
        failures=provider_failures,
        total_failed_providers=len(provider_failures),
        description="Provider failure details from the last automated test cycle (run every hour)",
    )


### UI routes

templates = Jinja2Templates(directory=TEMPLATES_PATH)


@router_ui.get("/")
def get_ui(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        name="index.html",
        request=request,
        context={
            "all_models": provider_and_models.all_model_names,
            "all_providers": [""] + provider_and_models.all_working_provider_names,
            "default_model": "gpt-4",
        },
    )


@router_ui.post("/completions")
async def get_completions(
    request: Request,
    payload: UiCompletionRequest,
    chat: type[g4f.ChatCompletion] = Depends(chat_completion),
) -> HTMLResponse:
    user_request = Message(role="user", content=payload.message)
    completion = await post_completion(
        CompletionRequest(messages=payload.history + [user_request]),
        CompletionParams(model=payload.model, provider=payload.provider),
        chat=chat,
    )
    bot_response = Message(role="assistant", content=completion.completion)
    return templates.TemplateResponse(
        name="messages.html",
        request=request,
        context={
            "messages": [user_request, bot_response],
        },
    )
