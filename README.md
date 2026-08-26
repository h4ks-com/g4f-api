# g4f-api

Alternative API for getting text completions from the [g4f project](https://github.com/xtekky/gpt4free/tree/main)

### Live at

- https://g4f.h4ks.com/

## Run

```bash
docker compose up --build
```

Or locally, reading settings from `.env` (see `.env.example`):

```bash
uv run python3 -m backend.run
```

Then make requests:

```bash
curl -X POST http://localhost:8000/api/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

Without `model` or `provider` query params the API picks one and falls back through others until it gets an answer. `GET /api/providers` and `GET /api/models` list what is currently available, and `GET /api/provider-failures` shows why the rest were dropped.

## Providers

Providers are discovered automatically at startup: every g4f provider that is marked working, needs no credentials, and never drives a headless browser. A background task probes them on boot and hourly, keeping only the ones that answer.

Probe them by hand with:

```bash
uv run python3 test_providers.py
```

## Tests

```bash
uv run python -m pytest -n 10 tests/
```
