# Reproducible container for the Streamlit app, using uv for fast, locked installs.
#   docker build -t bolt-calc .
#   docker run -p 8501:8501 bolt-calc
FROM python:3.12-slim

# Bring in the uv binary from its official (pinned) image.
COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /uvx /bin/

# Use the image's Python and a fixed venv location; never download a second Python.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install the exact locked runtime dependencies first (better layer caching, no dev tools).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
