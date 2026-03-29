import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.background import update_working_providers
from backend.errors import add_exception_handlers
from backend.routes import add_routers
from backend.settings import TEMPLATES_PATH, settings

logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.CHECK_WORKING_PROVIDERS:
        task = asyncio.create_task(_periodic_provider_check())
        yield
        task.cancel()
    else:
        yield


async def _periodic_provider_check() -> None:
    """Background task that checks working providers every hour."""
    await asyncio.sleep(2)
    try:
        await update_working_providers()
    except Exception:
        logging.exception("Initial provider check failed")

    while True:
        await asyncio.sleep(60 * 60)
        try:
            await update_working_providers()
        except Exception:
            logging.exception("Periodic provider check failed")


app = FastAPI(
    title="G4F API",
    description="Get text completions from various models and providers using https://github.com/xtekky/gpt4free",
    version="0.0.1",
    lifespan=lifespan,
)

add_exception_handlers(app)
add_routers(app)
app.mount("/static", StaticFiles(directory=TEMPLATES_PATH), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
__all__ = ["app"]
