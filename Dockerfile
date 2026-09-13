FROM python:3.14-slim-bookworm

ARG ATLAS_UID=1000
ARG ATLAS_GID=1000

COPY --from=ghcr.io/astral-sh/uv:0.11.5 /uv /uvx /bin/

ADD --chmod=644 https://repo.radeon.com/rocm/rocm.gpg.key /etc/apt/keyrings/rocm.asc
RUN echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.asc] https://repo.radeon.com/rocm/apt/7.2 jammy main' > /etc/apt/sources.list.d/rocm.list \
    && printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' > /etc/apt/preferences.d/rocm \
    && apt-get update \
    && apt-get install -y --no-install-recommends g++ hip-dev rccl-dev \
        hipcc=1.1.1.70200-43~22.04 rocm-device-libs=1.0.0.70200-43~22.04 \
    && rm -rf /var/lib/apt/lists/*

ADD https://github.com/ROCm/librocdxg/releases/download/v1.2.0/rocdxg-roct_1.2.0_amd64.deb /tmp/rocdxg.deb
# MIOpen compiles GRU kernels at runtime and needs C++ headers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libatomic1 libstdc++-12-dev libxml2 osmium-tool roctracer \
    && dpkg -i /tmp/rocdxg.deb \
    && rm -rf /var/lib/apt/lists/* /tmp/rocdxg.deb \
    && mkdir -p /opt/venv /tmp/uv-cache \
    && chown "$ATLAS_UID:$ATLAS_GID" /opt/venv /tmp/uv-cache

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src/ src/
COPY tests/ tests/
COPY .pre-commit-config.yaml .gitignore ./

USER ${ATLAS_UID}:${ATLAS_GID}

CMD ["uv", "run", "--locked", "atlas", "doctor"]
