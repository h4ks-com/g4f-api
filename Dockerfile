FROM python:3.12.2-slim-bullseye
RUN apt-get update && apt-get install -y curl gcc python3-dev
COPY --from=ghcr.io/astral-sh/uv:0.7.17 /uv /uvx /bin/
WORKDIR /backend
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
RUN export RUST_BACKTRACE=1 && uv sync

COPY backend/ backend
COPY static/ static


ENTRYPOINT ["uv", "run", "python3", "-m", "backend.run"]
