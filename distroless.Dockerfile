FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .

# Install build dependencies, including git for maxtext
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && \
    python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Final stage
FROM cgr.dev/chainguard/python:latest

WORKDIR /app
# Copy only the site-packages from the builder's virtual environment
# This allows the distroless image to use its default python entrypoint without PATH overrides
COPY --from=builder /opt/venv/lib/python3.12/site-packages /app/libs

COPY . .
# Point PYTHONPATH to both the installed libraries and the source code
ENV PYTHONPATH=/app/libs:/app/src

CMD ["-m", "gemma_4_sql.cli"]
