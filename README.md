# g4f-api

Alternative API for getting text completions from the [g4f project](https://github.com/xtekky/gpt4free/tree/main)

### Live at

- https://g4f.h4ks.com/

## Quick Start

```bash
./start.sh
```

Then make requests:

```bash
curl -X POST http://localhost:8001/api/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

## Provider Management

This API uses a **whitelist approach** for provider management. Only curated, working providers are enabled (see `PROVIDER_WHITELIST` in `backend/dependencies.py`). This ensures reliable responses and fast failures.

### Testing Providers

Test all whitelisted providers automatically:

```bash
uv run python3 test_providers.py
```

This will test each provider and show you which ones are currently working. Update `PROVIDER_WHITELIST` based on the results.
