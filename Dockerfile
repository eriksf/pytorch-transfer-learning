FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS base

# Install python and pip
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3.12-full \
        python3-pip \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:0.6.9 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY uv.lock pyproject.toml /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev
COPY images/ /app/images
COPY pytorch_transfer_learning /app/pytorch_transfer_learning
COPY LICENSE \
     README.md \
     /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM base
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"
# Build matpotlib font cache
RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot"

CMD [ "train", "--help" ]
