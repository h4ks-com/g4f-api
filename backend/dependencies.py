import inspect
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TypeVar

import aiohttp
import g4f
import g4f.cookies
from fastapi import Query
from fastapi.openapi.models import Example
from g4f.errors import G4FError
from g4f.models import ModelUtils
from g4f.Provider import BaseProvider
from g4f.Provider.base_provider import AsyncAuthedProvider
from g4f.providers.retry_provider import IterListProvider
from pydantic import BaseModel, Field

from backend.errors import CustomValidationError
from backend.models import CompletionModel, CompletionProvider, Message

logger = logging.getLogger(__name__)

# g4f otherwise harvests cookies from whatever browsers it finds on the host, which
# prompts for the macOS keychain locally and has nothing to read in a container.
g4f.cookies.BROWSERS = []

MODEL_BLACKLIST = [
    "TextGenerations",
    "ImageGenerations",
    "gpt-4o-mini-audio-preview",
]

# Names no provider currently serves are filtered at request time, so an entry that comes
# and goes upstream costs nothing.
BEST_MODELS_ORDERED = [
    "claude-opus-5",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
    "claude-opus-4.8",
    "gemini-3.7-flash",
    "grok-4.6",
    "gpt-5.5-pro",
    "claude-sonnet-5",
    "gpt-5.5",
    "claude-opus-4.7",
    "qwen3.8-max",
    "deepseek-v4-pro",
    "gpt-5.4-pro",
    "gemini-3.6-flash",
    "grok-4.5",
    "kimi-k3",
    "claude-opus-4.6",
    "gpt-5.4",
    "glm-5.3",
    "claude-sonnet-4.6",
    "gemini-3.5-flash",
    "gpt-5.3-chat",
    "deepseek-v4",
    "qwen3.7-max",
    "gpt-5.2",
    "gemini-3.1-pro",
    "claude-opus-4.5",
    "grok-4.1",
    "gpt-5.1",
    "kimi-k2.7",
    "deepseek-v3.2",
    "gpt-5",
    "o3-pro",
    "gemini-3-pro",
    "claude-sonnet-4.5",
    "qwen3-max",
    "o4-mini",
    "grok-3",
    "deepseek-v3",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4",
]

# Meta-providers, local runtimes and media-only endpoints: never remote text completions.
# Perplexity answers every prompt with the constant fragment "est.", which passes the
# background probe's non-empty check and then poisons automatic selection.
PROVIDER_BLACKLIST = {
    "AnyProvider",
    "CachedSearch",
    "Custom",
    "Ollama",
    "OpenAIFM",
    "Perplexity",
    "PollinationsAudio",
    "PollinationsImage",
}


def _provider_has_tools_support(provider: type[BaseProvider]) -> bool:
    """Check if a provider's source code references tool/function calling."""
    for method_name in (
        "create_async_generator",
        "create_generator",
        "create_completion",
    ):
        method = getattr(provider, method_name, None)
        if method is None:
            continue
        try:
            src = inspect.getsource(method)
        except (OSError, TypeError):
            continue
        if '"tools"' in src or "'tools'" in src:
            return True
    return False


def _is_media_only(provider: type[BaseProvider]) -> bool:
    models = set(getattr(provider, "models", None) or [])
    media = (
        set(getattr(provider, "image_models", None) or [])
        | set(getattr(provider, "audio_models", None) or [])
        | set(getattr(provider, "video_models", None) or [])
    )
    return bool(models) and models <= media


def _may_launch_browser(provider: type[BaseProvider]) -> bool:
    """Whether a provider can reach for a real browser.

    `use_nodriver` only covers providers that always need one, and a provider setting it
    False still drives a browser on its captcha path: Cloudflare declares False and dies
    on "Google Chrome / Chromium / Edge executable not found". The module referencing
    nodriver at all is the signal that holds.
    """
    if getattr(provider, "use_nodriver", False):
        return True
    module = sys.modules.get(provider.__module__)
    if module is None:
        return False
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):
        return False
    return "nodriver" in source or "zendriver" in source


def is_usable_provider(provider: type[BaseProvider]) -> bool:
    """A provider we can call straight from the server with no credentials.

    Excludes anything that may drive a headless browser or expects a logged-in session
    (`AsyncAuthedProvider`), since neither works in a container.
    """
    return (
        provider.working
        and not provider.needs_auth
        and not issubclass(provider, AsyncAuthedProvider)
        and not _may_launch_browser(provider)
        and not _is_media_only(provider)
        and provider.__name__ not in PROVIDER_BLACKLIST
    )


def discover_providers() -> dict[str, type[BaseProvider]]:
    return {
        provider.__name__: provider
        for provider in g4f.Provider.__providers__
        if is_usable_provider(provider)
    }


base_working_providers_map = discover_providers()


def _best_provider_names(
    best_provider: str | IterListProvider | type[BaseProvider],
) -> list[str]:
    """Provider names behind a model's `best_provider`, best first.

    g4f 8.x supplies a provider name, or an IterListProvider whose `.providers` are
    themselves names, ordered by preference.
    """
    if isinstance(best_provider, str):
        return [best_provider]
    if isinstance(best_provider, IterListProvider):
        return [
            p if isinstance(p, str) else p.__name__ for p in best_provider.providers
        ]
    return [best_provider.__name__]


def _coerce_model_names(models: object) -> set[str]:
    """Normalise the many shapes g4f providers use to declare models.

    Across providers this is variously a list of names, a list of {name: spec} dicts,
    a dict keyed by name, or absent entirely.
    """
    if isinstance(models, dict):
        return {str(k) for k in models}
    if not isinstance(models, (list, tuple, set)):
        return set()
    names: set[str] = set()
    for entry in models:
        if isinstance(entry, str):
            names.add(entry)
        elif isinstance(entry, dict):
            names.update(str(k) for k in entry)
    return names


def provider_model_names(provider: type[BaseProvider]) -> set[str]:
    """Every model name a provider declares, including its default and aliases."""
    names = _coerce_model_names(getattr(provider, "models", None))
    names |= _coerce_model_names(getattr(provider, "model_aliases", None))
    default_model = getattr(provider, "default_model", None)
    if isinstance(default_model, str) and default_model:
        names.add(default_model)
    return names


def load_provider_models(providers: dict[str, type[BaseProvider]]) -> None:
    """Ask providers that populate their model list lazily to fetch it.

    Most providers ship an empty `models` until `get_models()` does a network call, so
    without this they look like they support nothing at all.
    """

    def load(provider: type[BaseProvider]) -> None:
        get_models = getattr(provider, "get_models", None)
        if not callable(get_models) or _coerce_model_names(
            getattr(provider, "models", None)
        ):
            return
        try:
            get_models()
        except (G4FError, aiohttp.ClientError, OSError, ValueError, KeyError):
            logger.warning(
                "Could not load models for %s", provider.__name__, exc_info=True
            )

    with ThreadPoolExecutor(max_workers=16) as pool:
        pool.map(load, providers.values())


def model_default_providers(model_name: str) -> list[str]:
    """Providers g4f recommends for a model, best first, or empty when unknown.

    Resolved from `ModelUtils` rather than `g4f.get_model_and_provider`, which raises
    on the plain provider names g4f 8.x puts in `Model.best_provider`.
    """
    model = ModelUtils.convert.get(model_name)
    if model is None or not model.best_provider:
        return []
    return _best_provider_names(model.best_provider)


@dataclass
class ProviderAndModels:
    all_working_provider_names: list[str] = field(default_factory=list)
    all_working_providers_map: dict[str, CompletionProvider] = field(
        default_factory=dict
    )
    all_model_names: list[str] = field(default_factory=list)
    all_models_map: dict[str, CompletionModel] = field(default_factory=dict)

    def update_model_providers(
        self, working_providers_map: dict[str, BaseProvider]
    ) -> None:
        self.all_working_provider_names = list(working_providers_map.keys())
        self.all_working_providers_map = {}
        self.all_model_names = list(ModelUtils.convert.keys())
        self.all_models_map = {}

        for model_name in self.all_model_names:
            model = ModelUtils.convert[model_name]
            if not model.best_provider:
                continue
            best_providers = {
                model.base_provider,
                *_best_provider_names(model.best_provider),
            }

            complation_model = CompletionModel(
                name=model.name, supported_provider_names=best_providers
            )
            self.all_models_map[model_name] = complation_model

            # Populate providers with recomended models
            for provider_name in best_providers:
                if provider_name not in working_providers_map:
                    continue

                if provider_name not in self.all_working_providers_map:
                    self.all_working_providers_map[provider_name] = CompletionProvider(
                        name=provider_name,
                        supported_models=set(),
                        url=working_providers_map[provider_name].url or "",
                    )
                self.all_working_providers_map[provider_name].supported_models.add(
                    model.name
                )

        # Populate with models declared in the provider class definitions themselves
        for provider_name, provider in working_providers_map.items():
            if provider_name not in self.all_working_providers_map:
                self.all_working_providers_map[provider_name] = CompletionProvider(
                    name=provider_name,
                    supported_models=set(),
                    url=provider.url or "",
                )
            self.all_working_providers_map[provider_name].supported_models.update(
                provider_model_names(provider)
            )

            for model_name in self.all_working_providers_map[
                provider_name
            ].supported_models:
                if model_name not in self.all_models_map:
                    self.all_models_map[model_name] = CompletionModel(
                        name=model_name, supported_provider_names={provider_name}
                    )
                else:
                    self.all_models_map[model_name].supported_provider_names.add(
                        provider_name
                    )

        self.all_model_names = list(self.all_models_map.keys())

        # Detect tool support by inspecting provider source code
        for (
            provider_name,
            completion_provider,
        ) in self.all_working_providers_map.items():
            if provider_name in working_providers_map:
                provider = working_providers_map[provider_name]
                if _provider_has_tools_support(provider):
                    completion_provider.supports_tools = True


provider_and_models = ProviderAndModels()
# ponytail: import-time network fetch, ~2s, so routes and the OpenAPI schema see a
# populated model map immediately. Move into the lifespan task if importing offline
# (tests, tooling) or the delay before the server binds starts to matter.
load_provider_models(base_working_providers_map)
provider_and_models.update_model_providers(base_working_providers_map)


def _is_namespaced(model_name: str) -> bool:
    """Whether a name is provider- or community-scoped, e.g. `community/x/y`, `Groq:z`.

    These are duplicates or one-off remixes of plainly named models, so they belong at
    the back of the preference order.
    """
    return "/" in model_name or ":" in model_name


BEST_MODELS_ORDERED += sorted(
    (
        model_name
        for model_name in provider_and_models.all_model_names
        if model_name not in BEST_MODELS_ORDERED and model_name not in MODEL_BLACKLIST
    ),
    key=lambda name: (_is_namespaced(name), name),
)

# Position lookups run per candidate inside the retry loop, over thousands of entries.
MODEL_RANK = {model_name: rank for rank, model_name in enumerate(BEST_MODELS_ORDERED)}
UNRANKED_MODEL = len(BEST_MODELS_ORDERED)

A = TypeVar("A")


def generate_examples_from_values(values: list) -> dict[str, Example]:
    return {str(v or "--"): Example(value=v) for v in values}


def allowed_values_or_none(v: A | None, allowed: list[A]) -> A | None:
    if v is None:
        return v
    if v not in allowed:
        raise CustomValidationError(
            f"Value {v} not in allowed values: {allowed}", error={}
        )
    return v


class CompletionParams:
    def __init__(
        self,
        model: str | None = Query(
            None,
            description="LLM model to use for completion. If not specified, the best available model will be used.",
            openapi_examples=generate_examples_from_values(
                [None] + provider_and_models.all_model_names
            ),
        ),
        provider: str | None = Query(
            None,
            description="Provider to use for completion. If not specified, the best available provider will be used.",
            openapi_examples=generate_examples_from_values(
                [None] + provider_and_models.all_working_provider_names
            ),
        ),
    ):
        provider = provider or None
        model = model or None
        if not (model or provider):
            self.provider = None
            self.model = None
            return

        allowed_values_or_none(model, provider_and_models.all_model_names)
        allowed_values_or_none(provider, provider_and_models.all_working_provider_names)
        if model and provider:
            if provider not in provider_and_models.all_working_providers_map:
                raise CustomValidationError(
                    f"Provider {provider} not in working providers. Check available providers with /api/providers",
                    error={
                        "allowed_providers": provider_and_models.all_working_provider_names
                    },
                )
            provider_model = provider_and_models.all_working_providers_map[provider]
            if model not in provider_model.supported_models:
                raise CustomValidationError(
                    f"Model {model} not supported by provider {provider}. Check available providers and their supported models with /api/providers",
                    error={"allowed_models": list(provider_model.supported_models)},
                )

        self.provider = provider
        self.model = model


def chat_completion() -> type[g4f.ChatCompletion]:
    return g4f.ChatCompletion


class CompletionResponse(BaseModel):
    completion: str = Field(..., description="Completion of the messages")
    provider: str | None = Field(None, description="Provider used for completion")
    model: str | None = Field(None, description="Model used for completion")


class UiCompletionRequest(BaseModel):
    model: str | None = Field(
        None,
        description="Model to use for completion. If not specified, the best available model will be used.",
    )
    provider: str | None = Field(
        None,
        description="Provider to use for completion. If not specified, the best available provider will be used.",
    )
    message: str = Field(..., description="Current message from text input")
    history: list[Message] = Field(
        default_factory=list, description="History of past messages"
    )
