import asyncio
import logging
import traceback
from datetime import datetime, timedelta

from g4f import AsyncClient, ProviderType
from g4f.client.stubs import ChatCompletion

from backend.adapters import is_provider_refusal
from backend.dependencies import (
    base_working_providers_map,
    load_provider_models,
    provider_and_models,
    provider_model_names,
)
from backend.errors import CustomValidationError
from backend.models import ProviderFailure
from backend.settings import settings

FALLBACK_PROBE_MODEL = "gpt-4"

lock = asyncio.Lock()

provider_failures: dict[str, ProviderFailure] = {}

success_cache: list[tuple[str, str, datetime]] = []
SUCCESS_CACHE_TTL_MINUTES = 30


def add_successful_provider(provider_name: str, model: str) -> None:
    global success_cache
    current_time = datetime.now()

    success_cache = [
        entry
        for entry in success_cache
        if not (entry[0] == provider_name and entry[1] == model)
    ]

    success_cache.insert(0, (provider_name, model, current_time))
    _clean_expired_cache()


def get_cached_successful_providers(model_filter: str | None = None) -> list[str]:
    global success_cache
    _clean_expired_cache()

    seen_providers = set[str]()
    successful_providers = []

    for provider_name, model, _ in success_cache:
        if provider_name in seen_providers:
            continue
        if model_filter and model != model_filter:
            continue

        seen_providers.add(provider_name)
        successful_providers.append(provider_name)

    return successful_providers


def clear_success_cache() -> None:
    global success_cache
    success_cache = []


def _clean_expired_cache() -> None:
    global success_cache
    cutoff_time = datetime.now() - timedelta(minutes=SUCCESS_CACHE_TTL_MINUTES)
    success_cache = [entry for entry in success_cache if entry[2] > cutoff_time]


async def ai_respond(messages: list[dict], model: str, provider: ProviderType) -> str:
    """Generate a response from the AI."""
    client = AsyncClient()
    chat_completion: ChatCompletion = await client.chat.completions.create(
        messages=messages, model=model, provider=provider, stream=False
    )
    choices = chat_completion.choices
    if len(choices) == 0:
        raise CustomValidationError(
            "No response from the provider", error={"messages": messages}
        )

    return choices[0].message.content


def _probe_model(provider: ProviderType) -> str:
    """Pick the model most likely to answer when probing a provider."""
    default_model = getattr(provider, "default_model", None)
    if isinstance(default_model, str) and default_model:
        return default_model
    declared = provider_model_names(provider)
    if declared:
        return min(declared)
    known = provider_and_models.all_working_providers_map.get(provider.__name__)
    if known and known.supported_models:
        return min(known.supported_models)
    return FALLBACK_PROBE_MODEL


async def test_provider(
    provider: ProviderType, queue: asyncio.Queue, semaphore: asyncio.Semaphore
) -> bool:
    """Sends hi to a provider and check if there is response or error."""
    print(f"Testing provider {provider.__name__}")
    provider_name = provider.__name__
    messages = [{"role": "user", "content": "hi, how are you?"}]
    model = _probe_model(provider)

    async with semaphore:
        try:
            async with asyncio.timeout(settings.PROVIDER_TEST_TIMEOUT):
                text = await ai_respond(messages, model, provider=provider)
            result = bool(text.strip()) and not is_provider_refusal(text)

            # If successful, remove from failures store
            if result and provider_name in provider_failures:
                del provider_failures[provider_name]

        except ValueError as e:
            logging.exception(e)
            result = False
            provider_failures[provider_name] = ProviderFailure(
                error_type="ValueError",
                error_message=str(e),
                traceback=traceback.format_exc(),
                timestamp=datetime.now(),
                model_used=model,
                messages=messages,
                response=None,
            )
        except asyncio.TimeoutError as e:
            logging.exception(e)
            result = False
            provider_failures[provider_name] = ProviderFailure(
                error_type="TimeoutError",
                error_message=f"Request timed out after {settings.PROVIDER_TEST_TIMEOUT} seconds",
                traceback=traceback.format_exc(),
                timestamp=datetime.now(),
                model_used=model,
                messages=messages,
                response=None,
            )
        except Exception as e:
            logging.exception(e)
            result = False
            response_data = (
                getattr(e, "response", None) if hasattr(e, "response") else None
            )
            if response_data is not None and not isinstance(response_data, dict):
                try:
                    response_data = {
                        "status": getattr(response_data, "status", None),
                        "url": str(getattr(response_data, "url", "")),
                    }
                except Exception:
                    response_data = {"raw": str(type(response_data))}
            provider_failures[provider_name] = ProviderFailure(
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc(),
                timestamp=datetime.now(),
                model_used=model,
                messages=messages,
                response=response_data,
            )

        await queue.put((provider, result))
    return result


async def update_working_providers():
    if lock.locked():
        return

    async with lock:
        now_working_providers = set()
        queue = asyncio.Queue()
        providers = list(base_working_providers_map.values())
        await asyncio.to_thread(load_provider_models, base_working_providers_map)

        async def producer():
            semaphore = asyncio.Semaphore(8)
            await asyncio.gather(
                *[test_provider(provider, queue, semaphore) for provider in providers]
            )
            await queue.join()
            await queue.put((None, None))

        async def consumer():
            async with asyncio.timeout(5 * 60):
                while True:
                    provider, result = await queue.get()
                    if provider is None and result is None:
                        break
                    name = provider.__name__
                    if result:
                        now_working_providers.add(name)
                    queue.task_done()

        await asyncio.gather(producer(), consumer())

        print(
            f"Finished testing providers. Updating working providers to {len(now_working_providers)}"
        )
        provider_and_models.update_model_providers(
            {
                provider_name: base_working_providers_map[provider_name]
                for provider_name in now_working_providers
            }
        )

        # Clear success cache to start fresh after background testing
        clear_success_cache()
