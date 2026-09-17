# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.9.7 AS uv

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PRE_COMMIT_HOME=/opt/pre-commit \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        git \
        libatomic1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/

WORKDIR /workspace

# Separar las dependencias del código permite reutilizar esta capa mientras el
# lockfile no cambie.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-groups --no-install-project

COPY README.md .pre-commit-config.yaml ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-groups \
    && git init --quiet /tmp/precommit-bootstrap \
    && cp .pre-commit-config.yaml /tmp/precommit-bootstrap/ \
    && cd /tmp/precommit-bootstrap \
    && pre-commit install-hooks \
    && rm -rf /tmp/precommit-bootstrap

COPY notebooks ./notebooks

EXPOSE 8888 5000

CMD ["uv", "run", "--locked", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
