FROM python:3.12-slim

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install production dependencies only
RUN uv sync --no-dev --frozen

# Copy source code (excludes items in .dockerignore)
COPY agents/ ./agents/
COPY api/ ./api/
COPY core/ ./core/
COPY tools/ ./tools/
COPY config/ ./config/
COPY workflow/ ./workflow/
COPY main.py ./

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
