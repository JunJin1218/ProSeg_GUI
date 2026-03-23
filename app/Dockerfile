FROM python:3.11-slim

WORKDIR /workspace

# Install system dependencies for OpenCV and uv bootstrap.
RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /usr/local/bin/uv

# Copy dependency metadata first for better layer caching.
COPY pyproject.toml uv.lock ./
ARG UV_EXTRA=cpu
RUN uv sync --frozen --no-dev --extra ${UV_EXTRA}

# Copy the application source code and model weights after dependencies.
COPY app ./app
COPY MedSAM2_models ./MedSAM2_models
COPY checkpoint_10_MedSAM2_US_Heart.pt ./checkpoint_10_MedSAM2_US_Heart.pt

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
