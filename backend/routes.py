import logging
from functools import lru_cache

import g4f
import requests
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
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
    CompletionResponse,
    Message,
    UiCompletionRequest,
    chat_completion,
    provider_and_models,
)
from backend.errors import CustomValidationError
from backend.models import CompletionRequest, ProviderFailuresResponse
from backend.settings import TEMPLATES_PATH

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


class NofailParams:
    def __init__(self, model: str, provider: str):
        self.model = model
        self.provider = provider


def get_nofail_params(offset: int = 0) -> NofailParams:
    for model in BEST_MODELS_ORDERED:
        try:
            default_provider = g4f.get_model_and_provider(model, None, False)[1]
        except (ModelNotFoundError, ProviderNotWorkingError):
            logging.warning(f"Model not found or not working: {model}")
            continue

        if offset > 0:
            offset -= 1
            continue

        # Priority 1: Recently successful providers for this specific model
        for provider_name in get_cached_successful_providers(model_filter=model):
            if _is_provider_model_available(provider_name, model):
                return NofailParams(model=model, provider=provider_name)

        # Priority 2: Default provider if working
        if default_provider.__name__ in provider_and_models.all_working_provider_names:
            return NofailParams(model=model, provider=default_provider.__name__)

        # Priority 3: Any recently successful provider
        for provider_name in get_cached_successful_providers():
            if _is_provider_model_available(provider_name, model):
                return NofailParams(model=model, provider=provider_name)

        # Priority 4: Any working provider supporting the model
        for provider_name in provider_and_models.all_working_provider_names:
            if _is_provider_model_available(provider_name, model):
                return NofailParams(model=model, provider=provider_name)

    raise HTTPException(
        status_code=500, detail="Failed to find a model and provider to use"
    )


def _is_provider_model_available(provider_name: str, model: str) -> bool:
    return (
        provider_name in provider_and_models.all_working_provider_names
        and model
        in provider_and_models.all_working_providers_map[provider_name].supported_models
    )


def get_nofail_params_excluding_failed(
    failed_combinations: set[tuple[str, str]], offset: int = 0
) -> NofailParams:
    for model in BEST_MODELS_ORDERED:
        try:
            default_provider = g4f.get_model_and_provider(model, None, False)[1]
        except (ModelNotFoundError, ProviderNotWorkingError):
            logging.warning(f"Model not found: {model}")
            continue

        if offset > 0:
            offset -= 1
            continue

        # Priority 1: Recently successful providers for this specific model (excluding failed)
        for provider_name in get_cached_successful_providers(model_filter=model):
            if _is_provider_model_available(provider_name, model):
                if (model, provider_name) not in failed_combinations:
                    return NofailParams(model=model, provider=provider_name)

        # Priority 2: Default provider if working (excluding failed)
        if default_provider.__name__ in provider_and_models.all_working_provider_names:
            if (model, default_provider.__name__) not in failed_combinations:
                return NofailParams(model=model, provider=default_provider.__name__)

        # Priority 3: Any recently successful provider (excluding failed)
        for provider_name in get_cached_successful_providers():
            if _is_provider_model_available(provider_name, model):
                if (model, provider_name) not in failed_combinations:
                    return NofailParams(model=model, provider=provider_name)

        # Priority 4: Any working provider supporting the model (excluding failed)
        for provider_name in provider_and_models.all_working_provider_names:
            if _is_provider_model_available(provider_name, model):
                if (model, provider_name) not in failed_combinations:
                    return NofailParams(model=model, provider=provider_name)

    raise HTTPException(
        status_code=500, detail="Failed to find a model and provider to use"
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


@lru_cache(maxsize=1)
def get_public_ip() -> str | None:
    response = requests.get("https://api.ipify.org?format=json")
    if not response.ok:
        return None
    return response.json().get("ip")


@router_api.post("/completions")
def post_completion(
    completion: CompletionRequest,
    params: CompletionParams = Depends(),
    chat: type[g4f.ChatCompletion] = Depends(chat_completion),
) -> CompletionResponse:
    nofail = False
    if params.model is None:
        if params.provider is None:
            nofail_params = get_nofail_params()
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
        print(f"Trying model: {model_name} and provider: {provider_name}")
        try:
            response = chat.create(
                model=model_name,
                provider=provider_name,
                messages=[msg.model_dump() for msg in completion.messages],
                stream=False,
            )
            if isinstance(response, str):
                if response.strip() == "" and nofail:
                    failed_combinations.add((model_name, provider_name))
                    nofail_params = get_nofail_params_excluding_failed(
                        failed_combinations, attempt
                    )
                    model_name, provider_name = (
                        nofail_params.model,
                        nofail_params.provider,
                    )
                    continue

                completion_response = CompletionResponse(
                    completion=adapt_response(model_name, response),
                    model=model_name,
                    provider=provider_name,
                )

                # HACK: Workaround for IP ban from some providers
                ip = get_public_ip()
                if ip is not None and ip in response.lower():
                    if ip_detected_response is None:
                        ip_detected_response = completion_response
                    continue

                # Cache successful provider-model combination
                add_successful_provider(provider_name, model_name)
                return completion_response

            raise CustomValidationError(
                "Unexpected response type from g4f.ChatCompletion.create",
                error={"response": str(response)},
            )
        except Exception as e:
            if not nofail:
                raise e
            failed_combinations.add((model_name, provider_name))
            nofail_params = get_nofail_params_excluding_failed(
                failed_combinations, attempt
            )
            model_name, provider_name = nofail_params.model, nofail_params.provider

    # Better than nothing maybe
    if ip_detected_response is not None:
        # Cache this success too, as it did work despite IP detection
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
def get_completions(
    request: Request,
    payload: UiCompletionRequest,
    chat: type[g4f.ChatCompletion] = Depends(chat_completion),
) -> HTMLResponse:
    user_request = Message(role="user", content=payload.message)
    completion = post_completion(
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
