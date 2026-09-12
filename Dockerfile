FROM python:3.14-slim-bookworm

ARG ATLAS_UID=1000
ARG ATLAS_GID=1000

COPY --from=ghcr.io/astral-sh/uv:0.11.5 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src/ src/
COPY tests/ tests/
COPY .pre-commit-config.yaml .gitignore ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked \
    && chown -R "$ATLAS_UID:$ATLAS_GID" /opt/venv

USER ${ATLAS_UID}:${ATLAS_GID}

CMD ["uv", "run", "--locked", "atlas", "doctor"]
